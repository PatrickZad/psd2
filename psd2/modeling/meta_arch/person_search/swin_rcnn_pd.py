# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from psd2.config import configurable

from psd2.structures import ImageList, Instances, pairwise_iou, Boxes, BoxMode
from psd2.utils.events import get_event_storage
from psd2.layers import batched_nms
from .base import SearchBase

from ...proposal_generator import build_proposal_generator
from .. import META_ARCH_REGISTRY
from psd2.modeling.extend.solider import SideSwinTransformer
from psd2.modeling.id_assign import build_id_assigner
import copy
from psd2.modeling.roi_heads import build_roi_heads


@META_ARCH_REGISTRY.register()
class SwinF4RCNN(SearchBase):
    """
    stage3 w/ norm as output, semantic=0.6 all the time
    """

    @configurable
    def __init__(
        self,
        swin,
        proposal_generator,
        roi_heads,
        id_assigner,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        self.swin = swin
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.id_assigner = id_assigner
        for p in self.backbone.parameters():
            p.requires_grad = False
        for n, p in self.swin.named_parameters():  # norm2 is not supervised in solider
            if (
                "side_stages.0.downsample" not in n
                and "side_stages.1" not in n
                and "side_semantic_embed_w.0" not in n
                and "side_semantic_embed_b.0" not in n
                and "side_semantic_embed_w.1" not in n
                and "side_semantic_embed_b.1" not in n
                and not n.startswith("side_norm3")
                and not n.startswith("side_norm2")
            ):
                p.requires_grad = False

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.eval()
            self.swin.eval()
            self.swin.side_stages[-1].train()
            self.swin.side_stages[-2].downsample.train()
            self.swin.side_semantic_embed_w[-1].train()
            self.swin.side_semantic_embed_b[-1].train()
            self.swin.side_semantic_embed_w[-2].train()
            self.swin.side_semantic_embed_b[-2].train()
            self.swin.side_norm2.train()
            self.swin.side_norm3.train()
            self.swin.softplus.train()
            self.proposal_generator.train()
            self.roi_heads.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def load_state_dict(self, state_dict, strict):
        out = super().load_state_dict(state_dict, strict)
        with torch.no_grad():
            self.swin.load_side(state_dict)
        return out

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        # NOTE downsample module of stage3 is trainable
        swin = SideSwinTransformer(
            side_start_stage=3,
            semantic_weight=tr_cfg.SEMANTIC_WEIGHT,
            pretrain_img_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size
            if isinstance(patch_embed.patch_size, int)
            else patch_embed.patch_size[0],
            embed_dims=patch_embed.embed_dim,
            depths=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            window_size=tr_cfg.WIN_SIZE,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            with_cp=tr_cfg.WITH_CP,
        )
        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        roi_heads = SwinROIHeads(cfg, swin_out_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        ret.update(
            {
                "proposal_generator": build_proposal_generator(cfg, swin_out_shape),
                "roi_heads": roi_heads,
                "swin": swin,
            }
        )
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.swin, image_list, features, proposals, gt_instances
        )

        if self.training:
            losses.update(proposal_losses)
            assign_ids = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids[i]
            # nms
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i in pred_instances:
                pred_boxes_t = pred_i.pred_boxes.tensor
                pred_scores = pred_i.pred_scores
                filter_mask = pred_scores >= score_t
                pred_boxes_t = pred_boxes_t[filter_mask]
                pred_scores = pred_scores[filter_mask]
                cate_idx = pred_scores.new_zeros(
                    pred_scores.shape[0], dtype=torch.int64
                )
                # nms
                keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)
                pred_boxes_t = pred_boxes_t[keep]
                pred_scores = pred_scores[keep]
                pred_i.pred_boxes = Boxes(pred_boxes_t, BoxMode.XYXY_ABS)
                pred_i.pred_scores = pred_scores
                pred_i.reid_feats = pred_boxes_t  # trivial impl
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                pred_i.assign_ids = pred_i.assign_ids[filter_mask][keep]
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            # nms
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i, gt_i in zip(pred_instances, gt_instances):
                pred_boxes_t = pred_i.pred_boxes.tensor
                pred_scores = pred_i.pred_scores
                filter_mask = pred_scores >= score_t
                pred_boxes_t = pred_boxes_t[filter_mask]
                pred_scores = pred_scores[filter_mask]
                cate_idx = pred_scores.new_zeros(
                    pred_scores.shape[0], dtype=torch.int64
                )
                # nms
                keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)
                pred_boxes_t = pred_boxes_t[keep]
                pred_scores = pred_scores[keep]
                pred_i.pred_boxes = Boxes(pred_boxes_t, BoxMode.XYXY_ABS)
                pred_i.pred_scores = pred_scores
                pred_i.reid_feats = pred_boxes_t  # trivial impl
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances

    def swin_backbone(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()
        x = self.backbone(x)
        x = x[list(x.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        bonenum = 3
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum - 1:
                norm_layer = getattr(self.swin, f"side_norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return {"stage3": outs[-1]}


@META_ARCH_REGISTRY.register()
class SwinF4RCNN2(SwinF4RCNN):
    """
    stage3 w/o norm as output, semantic=1.0 for det
    """

    @classmethod
    def from_config(cls, cfg):
        ret = SearchBase.from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin = SideSwinTransformer(
            side_start_stage=3,
            semantic_weight=tr_cfg.SEMANTIC_WEIGHT,
            pretrain_img_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size
            if isinstance(patch_embed.patch_size, int)
            else patch_embed.patch_size[0],
            embed_dims=patch_embed.embed_dim,
            depths=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            window_size=tr_cfg.WIN_SIZE,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            with_cp=tr_cfg.WITH_CP,
        )
        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        swin_out_norm_shape = {
            "stage{}_norm".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        swin_out_shape.update(swin_out_norm_shape)
        roi_heads = SwinROIHeads2(cfg, swin_out_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        ret.update(
            {
                "proposal_generator": build_proposal_generator(cfg, swin_out_shape),
                "roi_heads": roi_heads,
                "swin": swin,
            }
        )
        return ret

    def swin_backbone(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()
        x = self.backbone(x)
        x = x[list(x.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        bonenum = 3
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum - 1:
                # norm_layer = getattr(self.swin, f'norm{i}')
                # out = norm_layer(out) NOTE this feat is to pass to the next stage, not the actual output
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return {"stage3": outs[-1]}


@META_ARCH_REGISTRY.register()
class SwinF4RCNN3(SwinF4RCNN2):
    """
    stage3 w/ norm as output, semantic=1.0 for det
    """

    def swin_backbone(self, x):  # keeps norm2
        return super(SwinF4RCNN2, self).swin_backbone(x)


@META_ARCH_REGISTRY.register()
class SwinF4RCNN4(SwinF4RCNN2):
    """
    stage3 w/ norm to rpn and w/o norm to later modules, semantic=1.0 for det
    """

    @classmethod
    def from_config(cls, cfg):
        assert (
            "norm" in cfg.MODEL.RPN.IN_FEATURES[-1]
        ), "Invalid MODEL.RPN.IN_FEATURES config"
        return super().from_config(cfg)

    def swin_backbone(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()
        x = self.backbone(x)
        x = x[list(x.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        outs_norm = []
        bonenum = 3
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum - 1:
                norm_layer = getattr(self.swin, f"side_norm{i}")
                out_norm = norm_layer(out)
                out_norm = (
                    out_norm.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs_norm.append(out_norm)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        return {"stage3_norm": outs_norm[-1], "stage3": outs[-1]}


@META_ARCH_REGISTRY.register()
class SwinF4RCNNSeq(SwinF4RCNN):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        swin = ret["swin"]
        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        roi_heads = SeqSwinROIHeads(cfg, swin_out_shape)
        ret["roi_heads"] = roi_heads
        return ret


from psd2.modeling.backbone.fpn import LastLevelMaxPool


@META_ARCH_REGISTRY.register()
class SwinSimFPNRCNN(SwinF4RCNN):
    @configurable
    def __init__(
        self,
        sim_fpn,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        self.sim_fpn = sim_fpn
        for p in self.sim_fpn.parameters():  # norm2 is not supervised in solider
            p.requires_grad = True

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        super().train(mode)
        if mode:
            # training:
            self.sim_fpn.train()

    @classmethod
    def from_config(cls, cfg):
        ret = super(SwinF4RCNN, cls).from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        # NOTE downsample module of stage3 is trainable
        swin = SideSwinTransformer(
            side_start_stage=3,
            semantic_weight=tr_cfg.SEMANTIC_WEIGHT,
            pretrain_img_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size
            if isinstance(patch_embed.patch_size, int)
            else patch_embed.patch_size[0],
            embed_dims=patch_embed.embed_dim,
            depths=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            window_size=tr_cfg.WIN_SIZE,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            with_cp=tr_cfg.WITH_CP,
            strides=(4, 2, 2, 1),  # NOTE remove last stride
        )

        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        sim_fpn_cfg = cfg.PERSON_SEARCH.DET.MODEL.SIM_FPN
        sim_fpn = SimpleFeaturePyramid(
            swin_out_shape,
            sim_fpn_cfg.IN_FEATURE,
            sim_fpn_cfg.OUT_CHANNELS,
            sim_fpn_cfg.SCALE_FACTORS,
            top_block=LastLevelMaxPool(),
        )
        roi_heads = AlteredStandaredROIHeads(cfg, sim_fpn.output_shape())
        ret["id_assigner"] = build_id_assigner(cfg)
        ret.update(
            {
                "proposal_generator": build_proposal_generator(
                    cfg, sim_fpn.output_shape()
                ),
                "roi_heads": roi_heads,
                "swin": swin,
                "sim_fpn": sim_fpn,
            }
        )
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        features = self.sim_fpn(features)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances
        )

        if self.training:
            losses.update(proposal_losses)
            """
            assign_ids = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )
            """

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = torch.zeros_like(
                    instances_i.scores, dtype=torch.int64
                )  # assign_ids[i]
            for pred_i in pred_instances:
                pred_scores = pred_i.scores
                pred_i.pred_scores = pred_scores
                pred_i.reid_feats = pred_i.pred_boxes.tensor  # trivial impl
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            for pred_i, gt_i in zip(pred_instances, gt_instances):
                pred_scores = pred_i.scores
                pred_i.pred_scores = pred_scores
                pred_i.reid_feats = pred_i.pred_boxes.tensor  # trivial impl
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances

    def swin_backbone(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()
        x = self.backbone(x)
        x = x[list(x.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        bonenum = 4
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum - 1:
                norm_layer = getattr(self.swin, f"side_norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return {"stage4": outs[-1]}


from psd2.modeling.roi_heads.roi_heads import (
    ROIHeads,
    add_ground_truth_to_proposals,
)
from psd2.modeling.poolers import ROIPooler
from psd2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from psd2.modeling.roi_heads import StandardROIHeads
from psd2.layers import ShapeSpec
from psd2.modeling.roi_heads.box_head import build_box_head


class AlteredStandaredROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if isinstance(pooler_resolution,int):
            pooler_resolution=(pooler_resolution,pooler_resolution)
        assert len(pooler_resolution) == 2
        assert isinstance(pooler_resolution[0], int) and isinstance(pooler_resolution[1], int)
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution[0],
                width=pooler_resolution[1],
            ),
        )
        box_predictor = FastRCNNOutputLayersPs(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }
    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        pos_match_indices = []  # for reid label assign
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            src_idxs = torch.arange(
                0,
                len(proposals_per_image),
                dtype=torch.int64,
                device=match_quality_matrix.device,
            )  # src_idxs are the indices after sampling
            tgt_idxs = matched_idxs[sampled_idxs]
            pos_mask = gt_classes < self.num_classes
            src_idxs = src_idxs[pos_mask]
            tgt_idxs = tgt_idxs[pos_mask]  # make it compatible with detr-like matchers
            pos_match_indices.append((src_idxs, tgt_idxs))

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt, pos_match_indices

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals, pos_match_indices = self.label_and_sample_proposals(
                proposals, targets
            )
        del targets

        if self.training:
            cur_pred, losses = self._forward_box_with_pred(features, proposals)

            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return cur_pred, losses, pos_match_indices
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, None
    def _forward_box_with_pred(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes: # NOTE not considered yet
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            with torch.no_grad():
                self.box_predictor.eval()
                pred_instances = self.box_predictor.inference_unms(predictions, proposals)
                self.box_predictor.train()
            return pred_instances, losses
        else:
            pred_instances = self.box_predictor.inference(predictions, proposals)
            return pred_instances,{}
    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:# NOTE not considered yet
                with torch.no_grad(): 
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances


class SwinROIHeads(ROIHeads):
    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        box_predictor: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        self.box_predictor = box_predictor
        self.mask_on = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        out_channels = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER.OUT_CHANNELS
        ret["box_predictor"] = FastRCNNOutputLayersPs(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        return ret

    def _shared_roi_transform(self, features, boxes, swin):
        x = self.pooler(features, boxes)
        if swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum = 3
        x, hw_shape = swin.side_stages[bonenum - swin.side_start_stage].downsample(
            x, hw_shape
        )
        if swin.semantic_weight >= 0:
            sw = swin.side_semantic_embed_w[bonenum - swin.side_start_stage](
                semantic_weight
            ).unsqueeze(1)
            sb = swin.side_semantic_embed_b[bonenum - swin.side_start_stage](
                semantic_weight
            ).unsqueeze(1)
            x = x * swin.softplus(sw) + sb
        for i, stage in enumerate(
            swin.side_stages[bonenum - swin.side_start_stage + 1 :]
        ):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if swin.semantic_weight >= 0:
                sw = swin.side_semantic_embed_w[
                    bonenum - swin.side_start_stage + 1 + i
                ](semantic_weight).unsqueeze(1)
                sb = swin.side_semantic_embed_b[
                    bonenum - swin.side_start_stage + 1 + i
                ](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i == len(swin.stages) - bonenum - 1:
                norm_layer = getattr(swin, f"side_norm{bonenum+i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, swin.num_features[bonenum + i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        return out

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        pos_match_indices = []  # for reid label assign
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            src_idxs = torch.arange(
                0,
                len(proposals_per_image),
                dtype=torch.int64,
                device=match_quality_matrix.device,
            )  # src_idxs are the indices after sampling
            tgt_idxs = matched_idxs[sampled_idxs]
            pos_mask = gt_classes < self.num_classes
            src_idxs = src_idxs[pos_mask]
            tgt_idxs = tgt_idxs[pos_mask]  # make it compatible with detr-like matchers
            pos_match_indices.append((src_idxs, tgt_idxs))

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt, pos_match_indices

    def forward(
        self,
        swin,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals, pos_match_indices = self.label_and_sample_proposals(
                proposals, targets
            )
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes, swin
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            with torch.no_grad():
                # for vis and id_assign
                self.box_predictor.eval()
                pred_instances = self.box_predictor.inference_unms(
                    predictions, proposals
                )
                self.box_predictor.train()
            del features
            return (
                pred_instances,
                losses,
                pos_match_indices,
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, {}, None

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            feature_list = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(
                feature_list, [x.pred_boxes for x in instances]
            )
            return self.mask_head(x, instances)
        else:
            return instances


class SeqSwinROIHeads(SwinROIHeads):
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.side_box_predictor = copy.deepcopy(self.box_predictor)

    def _shared_roi_transform_stage2(self, features, boxes, swin):
        x = self.pooler(features, boxes)
        if swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum = 3
        x, hw_shape = swin.stages[bonenum - 1].downsample(x, hw_shape)
        if swin.semantic_weight >= 0:
            sw = swin.semantic_embed_w[bonenum - 1](semantic_weight).unsqueeze(1)
            sb = swin.semantic_embed_b[bonenum - 1](semantic_weight).unsqueeze(1)
            x = x * swin.softplus(sw) + sb
        for i, stage in enumerate(swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if swin.semantic_weight >= 0:
                sw = swin.semantic_embed_w[bonenum + i](semantic_weight).unsqueeze(1)
                sb = swin.semantic_embed_b[bonenum + i](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i == len(swin.stages) - bonenum - 1:
                norm_layer = getattr(swin, f"norm{bonenum+i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, swin.num_features[bonenum + i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        return out

    def forward(
        self,
        swin,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        return_box_feat=False,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals, pos_match_indices = self.label_and_sample_proposals(
                proposals, targets
            )

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes, swin
        )
        predictions = self.side_box_predictor(
            box_features.mean(dim=[2, 3])
        )  # det only head
        if self.training:
            losses = self.side_box_predictor.losses(predictions, proposals)
            ls = list(losses.keys())
            for k in ls:
                losses[k + "_side"] = losses.pop(k)
        with torch.no_grad():
            # for second stage
            pred_instances = self.side_box_predictor.inference_unms(
                predictions, proposals
            )
        proposals = []
        for pred1 in pred_instances:
            res = Instances(pred1.image_size)
            res.proposal_boxes = pred1.pred_boxes
            res.objectness_logits = pred1.pred_scores
            proposals.append(res)
        if self.training:
            proposals, pos_match_indices = self.label_and_sample_proposals(
                proposals, targets
            )
        del targets
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform_stage2(
            [features[f] for f in self.in_features], proposal_boxes, swin
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        del features
        if self.training:
            losses2 = self.box_predictor.losses(predictions, proposals)
            losses.update(losses2)
            return (
                (
                    pred_instances,
                    losses,
                    pos_match_indices,
                    box_features,
                )
                if return_box_feat
                else (
                    pred_instances,
                    losses,
                    pos_match_indices,
                )
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return (
                pred_instances,
                {},
                None,
                box_features if return_box_feat else pred_instances,
                {},
                None,
            )


class SwinROIHeads2(SwinROIHeads):
    def _shared_roi_transform(self, features, boxes, swin):
        x = self.pooler(features, boxes)
        semantic_weight = 1.0  # this is only for detection, so give 1.0
        w = torch.ones(x.shape[0], 1) * semantic_weight
        w = torch.cat([w, 1 - w], axis=-1)
        semantic_weight = w.cuda()

        feat = x
        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum = 3
        x, hw_shape = swin.side_stages[bonenum - swin.side_start_stage].downsample(
            x, hw_shape
        )
        if swin.semantic_weight >= 0:
            sw = swin.side_semantic_embed_w[bonenum - swin.side_start_stage](
                semantic_weight
            ).unsqueeze(1)
            sb = swin.side_semantic_embed_b[bonenum - swin.side_start_stage](
                semantic_weight
            ).unsqueeze(1)
            x = x * swin.softplus(sw) + sb
        for i, stage in enumerate(
            swin.side_stages[bonenum - swin.side_start_stage + 1 :]
        ):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if swin.semantic_weight >= 0:
                sw = swin.side_semantic_embed_w[
                    bonenum - swin.side_start_stage + 1 + i
                ](semantic_weight).unsqueeze(1)
                sb = swin.side_semantic_embed_b[
                    bonenum - swin.side_start_stage + 1 + i
                ](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i == len(swin.stages) - bonenum - 1:
                norm_layer = getattr(swin, f"side_norm{bonenum+i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, swin.num_features[bonenum + i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        return out


class PromptedSwinROIHeads(SwinROIHeads):
    def _shared_roi_transform(
        self, features, boxes, swin, det_stage_prompts, task_query
    ):
        x = self.pooler(features, boxes)  # no prompt tokens
        if swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum = 3
        x, hw_shape = swin.side_stages[bonenum - swin.side_start_stage].downsample(
            x, hw_shape
        )
        if swin.semantic_weight >= 0:
            sw = swin.side_semantic_embed_w[bonenum - swin.side_start_stage](
                semantic_weight
            ).unsqueeze(1)
            sb = swin.side_semantic_embed_b[bonenum - swin.side_start_stage](
                semantic_weight
            ).unsqueeze(1)
            x = x * swin.softplus(sw) + sb
        # NOTE to simulate the added prompts in last stage
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(
            swin.side_stages[bonenum - swin.side_start_stage + 1 :]
        ):
            if (
                not isinstance(swin.num_prompts, int)
                and swin.num_prompts[i + bonenum] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
                task_query_stage = task_query_x.expand(-1, len(stage.blocks), -1)
                selected_prompts, p_loss = det_stage_prompts[i](
                    task_query_stage, f"det{i}", train=self.training
                )
                prompt_loss += p_loss
                expanded_prompts = []
                for bi in range(len(boxes)):
                    expanded_prompts.append(
                        selected_prompts[:, bi : bi + 1].expand(
                            -1, len(boxes[bi]), -1, -1
                        )
                    )
                selected_prompts = torch.cat(expanded_prompts, dim=1)
                del expanded_prompts
                x, hw_shape, out, out_hw_shape = stage(
                    x, hw_shape, deep_prompt_embd=selected_prompts
                )
            if swin.semantic_weight >= 0:
                sw = swin.side_semantic_embed_w[
                    bonenum - swin.side_start_stage + 1 + i
                ](semantic_weight).unsqueeze(1)
                sb = swin.side_semantic_embed_b[
                    bonenum - swin.side_start_stage + 1 + i
                ](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i == len(swin.stages) - bonenum - 1:
                norm_layer = getattr(swin, f"side_norm{bonenum+i}")
                # TODO check keep the output tokens of prompts
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, swin.num_features[bonenum + i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        if self.training:
            prompt_loss = {"loss_prompt_det": prompt_loss}
        return out, prompt_loss

    def forward(
        self,
        swin,
        det_stage_prompts,
        task_query,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals, pos_match_indices = self.label_and_sample_proposals(
                proposals, targets
            )
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features, prompt_loss = self._shared_roi_transform(
            [features[f] for f in self.in_features],
            proposal_boxes,
            swin,
            det_stage_prompts,
            task_query,
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            losses.update(prompt_loss)
            with torch.no_grad():
                # for vis and id_assign
                pred_instances = self.box_predictor.inference_unms(
                    predictions, proposals
                )
            del features
            return (
                pred_instances,
                losses,
                pos_match_indices,
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, {}, None


class FastRCNNOutputLayersPs(FastRCNNOutputLayers):
    def inference_unms(self, predictions, proposals):
        all_boxes = self.predict_boxes(predictions, proposals)
        all_scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        results = []
        # borrowed from fast_rcnn_inference
        for scores, boxes, image_shape in zip(all_scores, all_boxes, image_shapes):
            valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(
                dim=1
            )
            if not valid_mask.all():
                boxes = boxes[valid_mask]
                scores = scores[valid_mask]

            scores = scores[:, :-1]
            num_bbox_reg_classes = boxes.shape[1] // 4
            # Convert to Boxes to use the `clip` function ...
            boxes = Boxes(boxes.reshape(-1, 4))
            boxes.clip(image_shape)
            boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

            # 1. Filter results based on detection scores. It can make NMS more efficient
            #    by filtering out low-confidence detections.
            filter_mask = scores >= 0.0  # R x K NOTE disable this
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero()
            if num_bbox_reg_classes == 1:
                boxes = boxes[filter_inds[:, 0], 0]
            else:
                boxes = boxes[filter_mask]
            scores = scores[filter_mask]

            # 2. Apply NMS for each class independently. NOTE disable this

            result = Instances(image_shape)
            result.pred_boxes = Boxes(boxes)
            result.pred_scores = scores
            result.pred_classes = filter_inds[:, 1]
            results.append(result)
        return results


from psd2.modeling import Backbone
from psd2.layers import Conv2d, get_norm
import math


class SimpleFeaturePyramid(Backbone):
    """
    This module modifies SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        input_shapes,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        strides = [
            int(input_shapes[in_feature].stride / scale) for scale in scale_factors
        ]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: input feature map.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = x
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert (
            stride == 2 * strides[i - 1]
        ), "Strides {} {} are not log2 contiguous".format(stride, strides[i - 1])
