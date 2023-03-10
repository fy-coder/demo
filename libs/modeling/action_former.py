import math

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms

class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits

class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return out_offsets

class DecoupleNet(nn.Module):
    def __init__(
        self,
        input_dim,
        with_ln=False
    ):
        super().__init__()
        self.dim = input_dim // 2
        self.relu = nn.ReLU()
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(4):
            self.embd.append(
                MaskedConv1D(
                    self.dim, self.dim//2, 3,
                    stride=1, padding=1, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(self.dim//2))
            else:
                self.embd_norm.append(nn.Identity())
        
    def forward(self, feats, mask):
        flow = feats[:, :self.dim, :]
        rgb = feats[:, self.dim:, :]

        a, mask = self.embd[0](flow, mask)
        a = self.embd_norm[0](a)

        b, mask = self.embd[1](flow, mask)
        b = self.embd_norm[1](b)

        c, mask = self.embd[2](rgb, mask)
        c = self.embd_norm[2](c)

        d, mask = self.embd[3](rgb, mask)
        d = self.embd_norm[3](d)

        return torch.cat((a, b, c, d), dim=1)

class PtTransformer0(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg               # other cfg for testing
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, 6
        )]
        self.reg_range = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*6
        else:
            assert len(n_mha_win_size) == 6
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if True:
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : 512,
                    'n_head': 4,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : (2, 2, 5),
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : True,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': 512,
                    'n_embd_ks': embd_kernel_size,
                    'arch': (2, 2, 5),
                    'scale_factor': scale_factor,
                    'with_ln' : True
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            'identity',
            **{
                'in_channels' : [512] * 6,
                'out_channel' : 512,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : True
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            512, 512, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=True,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            512, 512, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=True
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        # self.decouple = DecoupleNet(2048)
        self.relu = nn.ReLU()

        self.scale = 1

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list, ref_model=None):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)  # [2, 2048, 2304]

        # batched_feats = self.decouple(batched_inputs, batched_masks)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        points = self.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # return loss during training
        if self.training:
            # permute the outputs
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks = [x.squeeze(1) for x in fpn_masks]

            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            
            time=1

            a, b = self.label_points(
                points, gt_segments, gt_labels, time)
            
            cls_loss = []
            reg_loss = []
            sco_loss = []
            for idx in range(time):
                gt_cls_labels = [a[i][idx] for i in range(len(a))]
                gt_offsets = [b[i][idx] for i in range(len(b))]

                # compute the loss and return
                loss, norm = self.losses(
                    fpn_masks,
                    out_cls_logits, out_offsets,
                    gt_cls_labels, gt_offsets, idx
                )
                cls_loss.append(loss['cls_loss'])
                reg_loss.append(loss['reg_loss'])
                sco_loss.append(loss['sco_loss'])

            # dcp_loss = self.dcp_loss(batched_feats, batched_masks) / norm

            cls_loss = torch.stack(cls_loss).mean()
            reg_loss = reg_loss[0]
            sco_loss = sco_loss[0]
            final_loss = cls_loss + reg_loss

            return {'cls_loss'   : cls_loss,
                    'reg_loss'   : reg_loss,
                    # 'sco_loss'   : sco_loss,
                    'final_loss' : final_loss}
        else:
            if ref_model != None:
                out_refines, out_probs, out_logits = ref_model(video_list)
                # print('eval_all')
            else:
                out_refines, out_probs, out_logits = None, None, None
                # print('eval_af')
            # permute the outputs
            # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
            out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
            # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
            fpn_masks = [x.squeeze(1) for x in fpn_masks]

            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets, out_refines, out_probs, out_logits
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def coarse_gt_single_video(self, gt_segment, gt_label, time=0, step=1, mode='none'):
        gt_label = gt_label.unsqueeze(1)
        base_segment = gt_segment
        base_label = gt_label

        if time == 0:
            seg = [base_segment]
            lab = [base_label.squeeze(1)]
            return seg, lab

        gt_segment = gt_segment.repeat(time, 1)
        gt_label = gt_label.repeat(time, 1)

        p_ctr = 0.4
        p_len = 0.4

        len = gt_segment[:, 1:] - gt_segment[:, :1]
        ctr = 0.5 * (gt_segment[:, :1] + gt_segment[:, 1:])

        d_ctr = (torch.rand(ctr.shape).to(ctr.device) * 2 - 1) * (p_ctr * len / 2)
        d_len = (torch.rand(len.shape).to(len.device) * 2 - 1) * (p_len * len)

        len += d_len
        ctr += d_ctr

        segment = torch.cat(((ctr-len/2).round(), (ctr+len/2).round()), dim=1)

        if mode == 'cat':
            segment = torch.cat((base_segment, segment), dim=0)
            label = torch.cat((base_label, gt_label), dim=0).squeeze(1)
        elif mode == 'list':
            segment = segment.reshape(time, -1, 2)
            label = gt_label.reshape(time, -1)
            seg = [base_segment]
            lab = [base_label.squeeze(1)]
            for idx in range(time):
                seg.append(segment[idx])
                lab.append(label[idx])
            return seg, lab

        return segment, label

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, time):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset, gt_refine, gt_prob = [], [], [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            # print(gt_segment.shape)
            coarse_segment, coarse_label = self.coarse_gt_single_video(gt_segment, gt_label, time=time-1, mode='list')
            # print(coarse_segment)
            # exit()
            aa = []
            bb = []
            for i, (a, b) in enumerate(zip(coarse_segment, coarse_label)):
                # print(a.shape)
                # print(b.shape)
                cls_targets, reg_targets = self.label_points_single_video(
                    concat_points, a, b
                )
                # append to list (len = # images, each of size FT x C)
                aa.append(cls_targets)
                bb.append(reg_targets)
            gt_cls.append(aa)
            gt_offset.append(bb)


        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            print('751')
            exit()
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        # print(reg_targets.shape)

        torch.set_printoptions(threshold=np.inf)
        # print(reg_targets.shape)
        # print(reg_targets[:20])
        reg_targets /= concat_points[:, 3, None]
        # print(reg_targets[:20])
        # print(gt_refine[:20])
        # print(gt_segment)
        # print('====================779')
        # exit()
        return cls_targets, reg_targets

    def dcp_loss(self, feats, masks):
        B, dim, T = feats.shape

        feats = feats.transpose(2, 1)
        masks = masks.transpose(2, 1)
        feats = feats.reshape(B*T, dim)
        masks = masks.reshape(B*T)

        feats = feats[masks]

        L = feats.shape[0]

        dim = feats.shape[1] // 2
        flow = feats[:, :dim]
        rgb = feats[:, dim:]

        dim = dim // 2
        flow_same = flow[:, :dim]
        flow_diff = flow[:, dim:]
        rgb_same = rgb[:, :dim]
        rgb_diff = rgb[:, dim:]

        cos_D1 = F.cosine_similarity(rgb_diff,rgb_same)
        cos_D2 = F.cosine_similarity(flow_diff,flow_same)
        cos_D3 = F.cosine_similarity(rgb_diff,flow_same)
        cos_D4 = F.cosine_similarity(flow_diff,rgb_same)

        cos_S1 = F.cosine_similarity(rgb_same,flow_same)
        cos_S2 = F.cosine_similarity(rgb_diff,flow_diff)

        loss_S = torch.mean((torch.ones(cos_S1.shape).to(cos_S1.device)-cos_S1)) \
                    +torch.mean((torch.ones(cos_S2.shape).to(cos_S2.device)-cos_S2)) 
        loss_D = torch.mean(torch.max(torch.zeros(cos_D1.shape).to(cos_S1.device),cos_D1)) \
                    +torch.mean(torch.max(torch.zeros(cos_D2.shape).to(cos_S1.device),cos_D2)) \
                    +torch.mean(torch.max(torch.zeros(cos_D3.shape).to(cos_S1.device),cos_D3)) \
                    +torch.mean(torch.max(torch.zeros(cos_D4.shape).to(cos_S1.device),cos_D4))

        ft_C = feats
        mean_C = torch.mean(ft_C, axis=0)
        var_C = torch.var(ft_C, axis=0)
        log_var_C = torch.log(var_C + 1e-6)
        loss_KL = torch.mean(mean_C*mean_C + var_C - log_var_C - 1) / 2
        loss = ((loss_S + loss_D)+0.1*loss_KL)
        # loss = loss_S + loss_D
        return (loss / max(L, 1)) * T

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, step
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # print(torch.stack(gt_offsets).shape)
        # print(torch.cat(out_offsets, dim=1).shape)
        # print(torch.stack(gt_offsets)[0, :100])
        # print(torch.cat(out_offsets, dim=1)[0, :100])
        # exit()

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)                                         # [2, 4536, 20]
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        if step == 0:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]
        # print(torch.cat(out_cls_logits, dim=1).shape)

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)  

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,                                          # [5053, 20]
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # print(pred_offsets)
            # print(gt_offsets)
            # exit()
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        # 3. score loss
        # sco_loss = score_loss(out_cls_logits, fpn_masks)
        # exit()
        # print('==========================')
        scores = []
        masks = []
        # poses = []

        tmp = torch.stack(gt_cls_labels).sum(-1) > 0  # [2, 4536]
        gt_labels = []
        s = 0
        l = 2304
        for i in range(6):
            gt_labels.append(tmp[:, s:s+l])
            s += l
            l = l // 2

        t = 1
        for i, (gt, cls_i, mask) in enumerate(zip(gt_labels, out_cls_logits, fpn_masks)):
            # print(gt.shape)
            pos = gt  # [2, 2304]
            mask = mask==False
            # print(mask.shape)
            # print(pos.shape)
            # exit()
            # print(cls_i)
            cls_i = torch.softmax(cls_i, dim=2)
            # cls_i = torch.sigmoid(cls_i)
            # print(cls_i)
            cls_i = torch.max(cls_i, dim=2).values
            # print(cls_i)
            cls_i[mask] = 1
            # print(cls_i)
            cls_i = cls_i.unsqueeze(2).expand(cls_i.shape[0], cls_i.shape[1], t).resize(cls_i.shape[0], 2304)
            mask = mask.unsqueeze(2).expand(mask.shape[0], mask.shape[1], t).resize(mask.shape[0], 2304)
            # pos = pos.unsqueeze(2).expand(pos.shape[0], pos.shape[1], t).resize(pos.shape[0], 2304)
            # print(cls_i.shape)
            # print(mask.shape)
            scores.append(cls_i)
            masks.append(mask)
            # poses.append(pos)
            t *= 2
            # print('---------------------------------------------------------------')
        
        scores = torch.stack(scores, dim=1)  # [2, 6, 2304]
        # print(scores)
        masks = torch.stack(masks, dim=1)
        # poses = torch.stack(poses, dim=1)

        # pos_idx = torch.sum(poses, dim=1) > 0

        idx = torch.sum(masks, dim=1)
        # idx = torch.logical_and(idx < 6, pos_idx)
        idx = idx < 6
        
        Low, low_idx = torch.min(scores, dim=1)
        # print((low_idx==0).sum())
        # print((low_idx==1).sum())
        # print((low_idx==2).sum())
        # print((low_idx==3).sum())
        # print((low_idx==4).sum())
        # print((low_idx==5).sum())
        # low_idx = (torch.ones(low_idx.shape, device=low_idx.device)*2).pow(low_idx)
        Low -= torch.ones(Low.shape, device=Low.device)*0.05  # 0.05
        # Low /= low_idx
        # exit()
        # scores[masks]=0
        # High = torch.max(scores, dim=1).values

        # print(low[0, :, 0])
        # low = torch.sort(low, dim=1).values
        # print(low[0, :, 0])
        # exit()

        low = Low[idx]
        # high = High[idx]

        # weight = (high + torch.ones(high.shape, device=high.device)*0.95)

        # low -= torch.ones(low.shape, device=low.device)*0.05  # 0.05

        # level = 3
        # low = low[:, :level, :]

        # sco_loss = low.sum() / (level*2304*low.shape[0])
        
        # sco_loss = (weight * low).sum()
        sco_loss = low.sum() 

        sco_loss /= idx.sum()
        # sco_loss /= self.loss_normalizer

        # t = High[torch.logical_and(idx < 6, pos_idx==False)]
        # print(t.shape)
        # idx = torch.randperm(t.shape[0])
        # t = t[idx].view(t.size())
        # print(t.shape)
        # print(low.shape)
        # sco_loss += t[:low.shape[0]*2].mean() - 0.05 - torch.log(scores[poses].mean())

        # sco_loss *= 0.1

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # sco_loss= sco_loss * max(num_pos, 1) / self.loss_normalizer

        # return a dict of losses
        # final_loss = cls_loss + reg_loss * loss_weight 

        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'sco_loss'   : sco_loss,}, self.loss_normalizer / max(num_pos, 1)

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets, out_refines, out_probs, out_logits
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            if out_refines == None:
                refines_per_vid, probs_per_vid, logits_per_vid = None, None, None
            else:
                refines_per_vid = [x[idx] for x in out_refines]
                probs_per_vid = [x[idx] for x in out_probs]
                logits_per_vid = [x[idx] for x in out_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]

            # for i in range(len(offsets_per_vid)):
            #     print(offsets_per_vid[i].shape)
            #     print(refines_per_vid[i].shape)

            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, refines_per_vid, probs_per_vid, logits_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        out_refines,
        out_probs,
        out_logits,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        if out_refines != None:
            for i, (cls_i, offsets_i, pts_i, mask_i) in enumerate(zip(
                    out_cls_logits, out_offsets, points, fpn_masks
                )):
                # sigmoid normalization for output logits
                pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
                pred_prob_max = (cls_i.softmax(1) * mask_i.unsqueeze(-1)).flatten()
                # print(cls_i.sigmoid().shape)
                # print(mask_i.unsqueeze(-1).shape)
                # print(pred_prob.shape)
                # print(pred_prob)
                
                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
                pred_prob = pred_prob[keep_idxs1]
                pred_prob_max = pred_prob_max[keep_idxs1]
                # print(pred_prob.shape)
                # print(pred_prob)
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                # print(pred_prob.shape)
                # print(pred_prob)
                # exit()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()
                pred_prob_max = pred_prob_max[idxs[:num_topk]].clone()

                # fix a warning in pytorch 1.9
                pt_idxs =  torch.div(
                    topk_idxs, self.num_classes, rounding_mode='floor'
                )
                cls_idxs = torch.fmod(topk_idxs, self.num_classes)

                # 3. gather predicted offsets
  
                offsets = offsets_i[pt_idxs]
                # print(offsets.shape)
                pts = pts_i[pt_idxs]

                # print(pred_prob)
                # print(cls_i.sigmoid()[pt_idxs, cls_idxs])

                # exit()
                # 4. compute predicted segments (denorm by stride for output offsets)
                seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
                seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]

                # pred_segs0 = torch.stack((seg_left, seg_right), -1)
                # # 5. Keep seg with duration > a threshold (relative to feature grids)
                # seg_areas = seg_right - seg_left
                # keep_idxs20 = seg_areas > self.test_duration_thresh

                pred_prob_len = max(pred_prob.max()-pred_prob.min(), 0.5)

                use_round = True  # -0.02 +0.04 -0.2 -0.56 -0.05 -0.02
                # if i!=0 and i!=1 and i!=2 and i!=3 and i!=4 and i!=5:
                # if i!=4 and i!=5:
                if True:  
                # if False:
                    # 0 1 2 3 4 5
                    a = [1,2,4,8,16,32,64,128]
                    b = 0  # 0
                    c = 1
                    e = 1
                    L = 5  # 5
                    f = 1  # 1
                    stride_i = a[min(i+b, L)]
                    s_i = a[i]
                    step = min(i+b+f, L+f)
                    for j in range(min(i+b+f, L+f)):  # 1 2 3 4 5 6
                        # if j>=5:
                        #     break
                        # 1 2 4 8 16 32
                        ref = out_refines[min(i+b, L)-j].squeeze(1)
                        prob = out_probs[min(i+b, L)-j].squeeze(1)
                        cls_ref = (torch.softmax(out_logits[min(i+b, L)-j].squeeze(1).reshape(prob.shape[0], 2, -1), dim=2)-0.05)*2
                        stride_j = a[min(i+b, L)-j]

                        lr = prob[:,0].max()-prob[:,0].min()+1e-6
                        rr = prob[:,1].max()-prob[:,1].min()+1e-6

                        t = min(i+b+1, L+1)

                        assert stride_i == stride_j

                        # c = max(s_i, stride_i)/s_i
                    
                        if use_round:
                            for e_ in range(e):
                                left_idx = (seg_left/stride_j).round().long()
                                right_idx = (seg_right/stride_j).round().long()

                                left_mask = torch.logical_and(left_idx >= 0, left_idx < 2304//stride_j)
                                right_mask = torch.logical_and(right_idx >= 0, right_idx < 2304//stride_j)

                                ref_left = ref[left_idx[left_mask], 0]  # todo
                                prob_left = prob[left_idx[left_mask], 0]
                                cls_left = cls_ref[:, 0, :][left_idx[left_mask], cls_idxs[left_mask]]
                                # print(cls_left.shape)
                                # print(pred_prob[left_mask].shape)
                                # print(cls_left)
                                # print(pred_prob[left_mask])
                                sp = 1
                                if i!=2 and i!=3:
                                    seg_left[left_mask] += (ref_left*stride_j/sp) * (1-pred_prob[left_mask])
                                else:
                                    seg_left[left_mask] += (ref_left*stride_j/c) * (1-pred_prob[left_mask])
                                    # seg_left[left_mask] += (ref_left*stride_j/c) * (1-pred_prob_max[left_mask])
                                    # seg_left[left_mask] += (ref_left*stride_j/c) * ((1-pred_prob_max[left_mask])+(1-pred_prob[left_mask]))/2
                                
                                ref_right = ref[right_idx[right_mask], 1]  # todo
                                prob_right = prob[right_idx[right_mask], 1] 
                                cls_right = cls_ref[:, 1, :][right_idx[right_mask], cls_idxs[right_mask]]
                                if i!=2 and i!=3:
                                    seg_right[right_mask] += (ref_right*stride_j/sp) * (1-pred_prob[right_mask])
                                else:
                                    seg_right[right_mask] += (ref_right*stride_j/c) * (1-pred_prob[right_mask])
                                    # seg_right[right_mask] += (ref_right*stride_j/c) * (1-pred_prob_max[right_mask])
                                    # seg_right[right_mask] += (ref_right*stride_j/c) * ((1-pred_prob_max[right_mask])+(1-pred_prob[right_mask]))/2
                                # seg_right[right_mask] += (ref_right*stride_j/c)
                                # seg_right[right_mask] += (ref_right*stride_j/c) * (1-pred_prob[right_mask]/pred_prob_len)

                                # cls_left[cls_left>0]=0
                                # cls_right[cls_right>0]=0
                                aa = cls_left*prob_left +1
                                bb = cls_right*prob_right +1
                                # aa = cls_left/stride_j +1
                                # bb = cls_right/stride_j +1
                                pred_prob[left_mask] *= aa
                                pred_prob[right_mask] *= bb
                        else:
                            left_idx0 = (seg_left/stride_j).floor().long()
                            left_idx1 = (seg_left/stride_j).ceil().long()
                            left_w1 = (seg_left/stride_j).frac()
                            right_idx0 = (seg_right/stride_j).floor().long()
                            right_idx1 = (seg_right/stride_j).ceil().long()
                            right_w1 = (seg_right/stride_j).frac()

                            left_mask = torch.logical_and(
                                            torch.logical_and(left_idx0 >= 0, left_idx0 < 2304//stride_j),
                                            torch.logical_and(left_idx1 >= 0, left_idx1 < 2304//stride_j))
                            right_mask = torch.logical_and(
                                            torch.logical_and(right_idx0 >= 0, right_idx0 < 2304//stride_j),
                                            torch.logical_and(right_idx1 >= 0, right_idx1 < 2304//stride_j))

                            ref_left0 = ref[left_idx0[left_mask], 0] 
                            ref_left1 = ref[left_idx1[left_mask], 0] 
                            w1 = left_w1[left_mask]
                            
                            # prob_left = prob[left_idx[left_mask], 0]
                            ref_left = ref_left0 * (1 - w1) + ref_left1 * w1
                            # print(left_idx0[left_mask])
                            # print(left_idx1[left_mask])
                            # print(ref_left0)
                            # print(ref_left1)
                            # print(w1)
                            # print(ref_left)
                            # exit()
                            seg_left[left_mask] += (ref_left * stride_j / c) * (1 - pred_prob[left_mask])
                            # seg_left[left_mask] += (ref_left * stride_j / c)
                            # * (1 - pred_prob[left_mask])
                            # print(ref_left*stride_j)
                            ref_right0 = ref[right_idx0[right_mask], 1] 
                            ref_right1 = ref[right_idx1[right_mask], 1] 
                            w1 = right_w1[right_mask]
                            # prob_right = prob[right_idx[right_mask], 1]
                            ref_right = ref_right0 * (1 - w1) + ref_right1 * w1
                            seg_right[right_mask] += (ref_right * stride_j / c) * (1 - pred_prob[right_mask])
                            # seg_right[right_mask] += (ref_right * stride_j / c)

                        stride_i //= 2
                        

                if True:
                    pred_segs = torch.stack((seg_left, seg_right), -1)
                    # 5. Keep seg with duration > a threshold (relative to feature grids)
                    seg_areas = seg_right - seg_left
                    keep_idxs2 = seg_areas > self.test_duration_thresh

                    # *_all : N (filtered # of segments) x 2 / 1
                    segs_all.append(pred_segs[keep_idxs2])
                    scores_all.append(pred_prob[keep_idxs2])
                    cls_idxs_all.append(cls_idxs[keep_idxs2])
                else:
                    pred_segs = torch.stack((seg_left, seg_right), -1)
                    # 5. Keep seg with duration > a threshold (relative to feature grids)
                    seg_areas = seg_right - seg_left
                    keep_idxs2 = seg_areas > self.test_duration_thresh

                    # *_all : N (filtered # of segments) x 2 / 1
                    # segs_all.append(torch.cat((pred_segs[keep_idxs2], pred_segs0[keep_idxs20]), dim=0))
                    # scores_all.append(torch.cat((pred_prob[keep_idxs2], pred_prob[keep_idxs20]), dim=0))
                    # cls_idxs_all.append(torch.cat((cls_idxs[keep_idxs2], cls_idxs[keep_idxs20]), dim=0))
                    segs_all.append(pred_segs[keep_idxs2])
                    scores_all.append(pred_prob[keep_idxs2])
                    cls_idxs_all.append(cls_idxs[keep_idxs2])
                    segs_all.append(pred_segs0[keep_idxs20])
                    scores_all.append(pred_prob[keep_idxs20])
                    cls_idxs_all.append(cls_idxs[keep_idxs20])
            
        else:
            for i, (cls_i, offsets_i, pts_i, mask_i) in enumerate(zip(
                    out_cls_logits, out_offsets, points, fpn_masks
                )):
                # sigmoid normalization for output logits
                pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
                
                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
                pred_prob = pred_prob[keep_idxs1]
                
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # fix a warning in pytorch 1.9
                pt_idxs =  torch.div(
                    topk_idxs, self.num_classes, rounding_mode='floor'
                )
                cls_idxs = torch.fmod(topk_idxs, self.num_classes)

                # 3. gather predicted offsets
                offsets = offsets_i[pt_idxs]
                pts = pts_i[pt_idxs]

                # 4. compute predicted segments (denorm by stride for output offsets)
                seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
                seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]

                pred_segs = torch.stack((seg_left, seg_right), -1)
                
                # 5. Keep seg with duration > a threshold (relative to feature grids)
                seg_areas = seg_right - seg_left
                keep_idxs2 = seg_areas > self.test_duration_thresh

                # *_all : N (filtered # of segments) x 2 / 1
                segs_all.append(pred_segs[keep_idxs2])
                scores_all.append(pred_prob[keep_idxs2])
                cls_idxs_all.append(cls_idxs[keep_idxs2])
        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}
        # print('===================1115')
        # exit()
        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                # print(segs)
                # print(stride)
                # print(nframes)
                segs = (segs * stride + 0.5 * nframes) / fps
                # print(segs*fps)
                # exit()
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results
