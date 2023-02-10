import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm

from ..utils import batched_nms
import numpy as np
from .losses import sigmoid_focal_loss

class Refinement_module(nn.Module):

    def __init__(
            self,
            backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines #layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_dim,  # input feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attache layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            train_cfg,  # other cfg for training
            test_cfg  # other cfg for testing
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor ** i for i in range(
            fpn_start_level, backbone_arch[-1] + 1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
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
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch': backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln,
                    'attn_pdrop': 0.0,
                    'proj_pdrop': self.train_dropout,
                    'path_pdrop': self.train_droppath,
                    'use_abs_pe': use_abs_pe,
                    'use_rel_pe': use_rel_pe
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'start_level': fpn_start_level,
                'with_ln': fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_strides': self.fpn_strides,
                'regression_range': self.reg_range
            }
        )
        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.test_loss = nn.MSELoss()
        # refinement新增
        self.refineHead = RefineHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )
        # self.clsHead = ClsHead(
        #     fpn_dim, head_dim, len(self.fpn_strides),
        #     kernel_size=head_kernel_size,
        #     num_layers=head_num_layers,
        #     with_ln=head_with_ln
        # )

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        points = self.point_generator(fpn_feats)

        out_refines, out_probs, out_logits = self.refineHead(fpn_feats, fpn_masks)
        # out_logits = self.clsHead(fpn_feats, fpn_masks)
        
        out_refines = [x.permute(0, 2, 1) for x in out_refines]
        out_probs = [x.permute(0, 2, 1) for x in out_probs]
        out_logits = [x.permute(0, 2, 1) for x in out_logits]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        if self.training:
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            time_ = 1
            gt_ref_low, gt_ref_high, gt_cls = self.label_points(
                points, gt_segments, gt_labels, time_
            )
            ref_loss = []
            inf_loss = []
            prob_loss = []
            cls_loss = []
            for idx in range(time_):
                a = [gt_ref_low[i][idx] for i in range(len(gt_ref_low))]
                b = [gt_ref_high[i][idx] for i in range(len(gt_ref_high))]
                c = [gt_cls[i][idx] for i in range(len(gt_cls))]

                # compute the loss and return
                loss = self.losses(
                    fpn_masks,
                    out_refines,
                    out_probs,
                    out_logits,
                    a, b, c, idx
                )
                ref_loss.append(loss['ref_loss'])
                inf_loss.append(loss['inf_loss'])
                prob_loss.append(loss['prob_loss'])
                cls_loss.append(loss['cls_loss'])

            ref_loss = torch.stack(ref_loss).min()*2  # 2
            inf_loss = torch.stack(inf_loss).min()*0
            prob_loss = torch.stack(prob_loss).min()*0.5  # 0.5

            cls_loss = torch.stack(cls_loss).min()*1
            final_loss = ref_loss  + prob_loss + cls_loss

            return {
                    'ref_loss': ref_loss,
                    # 'inf_loss': inf_loss,
                    'prob_loss': prob_loss,
                    'cls_loss': cls_loss,
                    'final_loss': final_loss
            }
        else:
            return out_refines, out_probs, out_logits

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
    def label_points(self, points, gt_segments, gt_labels, time_):
        concat_points = torch.cat(points, dim=0)
        gt_ref_low, gt_ref_high, gt_cls = [], [], []
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            coarse_segments, coarse_labels = self.coarse_gt_single_video(
                gt_segment, gt_label, time=time_ - 1, mode='list'
            )
            gt_ref_low_single = []
            gt_ref_high_single = []
            gt_cls_single = []
            for i, (coarse_segment, coarse_label) in enumerate(zip(coarse_segments, coarse_labels)):
                
                low_targets, high_targets, cls_targets = \
                    self.label_points_single_video(
                        concat_points, coarse_segment, coarse_label
                    )
                # append to list (len = # images, each of size FT x C)
                gt_ref_low_single.append(low_targets)
                gt_ref_high_single.append(high_targets)
                gt_cls_single.append(cls_targets)
            gt_ref_low.append(gt_ref_low_single)
            gt_ref_high.append(gt_ref_high_single)
            gt_cls.append(gt_cls_single)

        return gt_ref_low, gt_ref_high, gt_cls

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # refine gt [4536]
        lis = concat_points[:, 0].long()
        pt = concat_points[:, :1, None]  # [4536, 1, 1]
        pt = pt.expand(num_pts, num_gts, 2)  # [4536, N, 2]
        gt = gt_segment[None].expand(num_pts, num_gts, 2)  # [4536, N, 2]
        dis = gt - pt  # [4536, N, 2]  左：+, 右：-
        abs_dis = torch.abs(dis)
        dis0, dis_idx1 = torch.min(abs_dis, dim=1)  # [4536, N, 2] -> [4536, 2]
        dis_idx0 = dis_idx1.long()  # [4536, 2]
        
        gt_ref_low = dis0.clone()
        gt_ref_high = dis0.clone()

        low_p = 0.2  # 0 ~ 1
        high_p = 0.2

        ra = concat_points[:, 1]
        rb = concat_points[:, 2]
        r = concat_points[0, 2]

        label = gt_label[None, :, None].expand(num_pts, num_gts, 2)  # [4536, N, 2]
        label = label.transpose(2, 1)[lis[:, None].repeat(1, 2), lis[:2][None, :].repeat(num_pts, 1), dis_idx0]  # [4536, 2]
        hot = torch.zeros((num_pts, 2, 20), device=label.device)
        hot[lis[:, None].repeat(1, 2), lis[:2][None, :].repeat(num_pts, 1), label] = 1
        # torch.set_printoptions(threshold=np.inf)
        # print(label[0,0])
        # print(hot[0,0,:])
        # print(hot.shape)
        # exit()
        
        # print(gt_label[0])
        # print(label[:10])
        # exit()

        for i in range(2):
            dis_l = gt_ref_low[:, i]
            dis_h = gt_ref_high[:, i]
            # F T
            range_out = torch.logical_and(torch.logical_and(
                (dis_l >= ra),
                (dis_l <= rb)
            ), (dis_l > r))
            range_in = torch.logical_or((dis_l < ra), (dis_l <= r))
            range_inf = (dis_l > rb)

            dis_l /= concat_points[:, 3]  # 0 ~ 1
            dis_h /= concat_points[:, 3]
            # print(dis_l[2303:2323])
            # print(dis_h[2303:2323])

            dis_h[range_in] = dis_h[range_in] * (1 + high_p)
            dis_l[range_in] = dis_l[range_in] * (1 - low_p)
            # print(dis_l[2303:2323])
            # print(dis_h[2303:2323])

            dis_h[range_out] += (r/2) * high_p
            dis_l[range_out] -= (r/2) * low_p
            # print(dis_l[2303:2323])
            # print(dis_h[2303:2323])

            # dis_l[dis_l>1] = 1
            # dis_h[dis_h>1] = 1

            dis_l.masked_fill_(range_inf==1, float('inf'))
            dis_h.masked_fill_(range_inf==1, float('inf'))
            # print(dis_l[2303:2323])
            # print(dis_h[2303:2323])
            # print('-------------')
        # exit()
        idx = dis.transpose(2, 1)[lis[:, None].repeat(1, 2), lis[:2][None, :].repeat(num_pts, 1), dis_idx0] < 0
        gt_ref_low[idx] *= -1
        gt_ref_high[idx] *= -1

        # exit()

        return gt_ref_low, gt_ref_high, hot

    def losses(
            self, fpn_masks,
            out_refines, out_probs, out_logits,
            gt_ref_low, gt_ref_high, gt_cls, step
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1 ref_loss
        gt_low = torch.stack(gt_ref_low)
        gt_high = torch.stack(gt_ref_high)
        gt_cls = torch.stack(gt_cls)
        out_ref = torch.cat(out_refines, dim=1).squeeze(2)  # [B, 4536, 2]   
        out_prob = torch.cat(out_probs, dim=1).squeeze(2)  
        a, b, c = out_prob.shape 
        out_logit = torch.cat(out_logits, dim=1).squeeze(2).reshape(a, b, c, -1)  # [B, 4536, 2, 20]

        if step == 0:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            )

        # t = 0
        # refs = []
        # masks = []
        # a = [1, 2, 4, 8, 16, 32]
        # for i, ref_i in enumerate(out_refines):
        #     B, T, C = ref_i.shape
        #     mask = torch.isinf(gt_low[:, t:t+T, :])==False
        #     v_mask = valid_mask[:, t:t+T, None].repeat(1, 1, 2)
        #     mask = torch.logical_and(mask, v_mask)
        #     ref = (ref_i*a[i])[:, :, None, :].repeat(1, 1, a[i], 1).reshape(B, -1, C)[:, None, :, :]
        #     mask = mask[:, :, None, :].repeat(1, 1, a[i], 1).reshape(B, -1, C)[:, None, :, :]
        #     refs.append(ref)
        #     masks.append(mask)
        #     t += T
        # refs = torch.cat(refs, dim=1)
        # masks = torch.cat(masks, dim=1)
        # print(refs[0,:,:10,0])
        # refs[masks==False] = 0
        # print(refs[0,:,:10,0])
        # cnt = masks.sum(dim=1)
        # # print(cnt)
        # mean = (refs.sum(dim=1)/cnt)[:, None, :, :].repeat(1, 6, 1, 1)
        # c_loss = torch.abs(refs[masks]-mean[masks]).mean()
        # print(c_loss)
        # exit()
          

        outside = torch.isinf(gt_low)
        valid = valid_mask[:, :, None].repeat(1, 1, 2)
        mask = torch.logical_and((outside == False), valid)
        out_mask = torch.logical_and((outside == True), valid)

        inf_loss = F.smooth_l1_loss(out_ref[out_mask], out_ref[out_mask]*0, reduction='mean')
        gt_prob = torch.ones(outside.shape, device=outside.device)
        gt_prob[outside] = 0
        prob_loss = F.smooth_l1_loss(out_prob[valid], gt_prob[valid], reduction='mean')

        gt_low = gt_low[mask]
        out_ref = out_ref[mask]
        gt_high = gt_high[mask]

        a = out_ref - gt_low
        b = out_ref - gt_high
        mask_in = (a * b) < 0

        c = torch.cat((torch.abs(a)[:, None], torch.abs(b)[:, None]), dim=-1)
        dis = torch.mean(c, dim=-1)

        # dis, _ = torch.min(c, dim=-1)

        # dis[mask_in] = 0  # dont!
        # ref_loss = dis.mean()
        ref_loss = dis[mask_in==False].mean()

        # ref_loss /= self.loss_normalizer
        # inf_loss /= self.loss_normalizer

        # 2. cls_loss
        # print(out_logit.shape)
        # print(gt_cls.shape)
        # print(out_logit[0,0,0])
        # print(gt_cls[0,0,0])

        # print(out_logit[mask].shape)
        # print(gt_cls[mask].shape)
        # gt_cls[out_mask] *= 0
        gt_target = gt_cls[mask]
        # gt_target *= 1 - self.train_label_smoothing
        # gt_target += self.train_label_smoothing / (self.num_classes + 1)
        # print(out_logit[mask][:10])
        # print(gt_target[:10])
        # exit()
        num_pos = mask.sum()
        if step == 0:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)

        cls_loss = sigmoid_focal_loss(
            out_logit[mask],
            gt_target,                                          # [3011, 20]
            reduction='sum'
        ) / self.loss_normalizer

        # cls_loss = F.smooth_l1_loss(
        #     out_logit[mask],
        #     gt_target,                                          # [3011, 20]
        #     reduction='mean'
        # )
        # exit() 
        return {
                'ref_loss': ref_loss,
                'inf_loss': inf_loss,
                'cls_loss': cls_loss,
                'prob_loss'  : prob_loss,
        }

    @torch.no_grad()
    def postprocessing(self, results):
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
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen

            # 4: repack the results
            processed_results.append(
                {'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels}
            )

        return processed_results

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

        p_ctr = 0.1
        p_len = 0.1

        len = gt_segment[:, 1:] - gt_segment[:, :1]
        ctr = 0.5 * (gt_segment[:, :1] + gt_segment[:, 1:])

        d_ctr = (torch.rand(ctr.shape).to(ctr.device) * 2 - 1) * (p_ctr * len / 2)
        d_len = (torch.rand(len.shape).to(len.device) * 2 - 1) * (p_len * len)

        len += d_len
        ctr += d_ctr

        segment = torch.cat(((ctr - len / 2).round(), (ctr + len / 2).round()), dim=1)

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


class ClsHead(nn.Module):
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
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
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
                    padding=kernel_size // 2,
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
        self.cls_head = MaskedConv1D(
            feat_dim, 40, kernel_size,
            stride=1, padding=kernel_size // 2
        )
        prior_prob = 0.01
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

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


class RefineHead(nn.Module):
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
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
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
                    padding=kernel_size // 2,
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
            stride=1, padding=kernel_size // 2
        )
        self.prob_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )
        self.cls_head = MaskedConv1D(
            feat_dim, 40, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        out_probs = tuple()
        out_logits = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            cur_probs, _ = self.prob_head(cur_out, cur_mask)
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_offsets += (self.scale[l](cur_offsets),)
            out_probs += (torch.sigmoid(cur_probs),)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_offsets, out_probs, out_logits

