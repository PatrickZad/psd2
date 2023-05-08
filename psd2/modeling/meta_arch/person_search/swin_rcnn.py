# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional
import torch
from torch import nn
import torch.nn.functional as tF
from psd2.config import configurable

from psd2.structures import ImageList, Instances,pairwise_iou,Boxes, BoxMode
from psd2.utils.events import get_event_storage
from psd2.layers import  batched_nms
from .base import SearchBase

from ...proposal_generator import build_proposal_generator
from .. import META_ARCH_REGISTRY
from psd2.modeling.extend.solider import SwinTransformer,PromptedSwinTransformer
from psd2.modeling.id_assign import build_id_assigner


@META_ARCH_REGISTRY.register()
class SwinF4RCNN(SearchBase):


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

        super().__init__(*args,**kws,)
        self.swin= swin
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.id_assigner=id_assigner
        for p in self.backbone.parameters():
            p.requires_grad=False
        for n,p in self.swin.named_parameters(): # norm2 is not supervised in solider
            if "stages.2.downsample" not in n and "stages.3" not in n and  "semantic_embed_w.3" not in n and  "semantic_embed_b.3" not in n and  "semantic_embed_w.2" not in n and  "semantic_embed_b.2" not in n and not n.startswith("norm3") and not n.startswith("norm2"): 
                p.requires_grad=False
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training=mode
        if mode:
            # training:
            self.backbone.eval()
            self.swin.eval()
            self.swin.stages[-1].train()
            self.swin.stages[-2].downsample.train()
            self.swin.semantic_embed_w[-1].train()
            self.swin.semantic_embed_b[-1].train()
            self.swin.semantic_embed_w[-2].train()
            self.swin.semantic_embed_b[-2].train()
            self.swin.norm2.train()
            self.swin.norm3.train()
            self.swin.softplus.train()
            self.proposal_generator.train()
            self.roi_heads.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin=SwinTransformer(semantic_weight=tr_cfg.SEMANTIC_WEIGHT,
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
            drop_path_rate=tr_cfg.DROP_PATH,)
        swin_out_shape={"stage{}".format(i+1): ShapeSpec(channels=swin.num_features[i],stride=swin.strides[i]) for i in range(len(swin.stages))}
        roi_heads=SwinROIHeads(cfg,swin_out_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        ret.update({
            "proposal_generator": build_proposal_generator(
                cfg, swin_out_shape
            ),
            "roi_heads": roi_heads,
            "swin": swin,
        })
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features=self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.swin,image_list, features, proposals, gt_instances
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
                pred_i.reid_feats = pred_boxes_t # trivial impl
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                pred_i.assign_ids=pred_i.assign_ids[filter_mask][keep]
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            # nms
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i, gt_i in zip(
                pred_instances, gt_instances
            ):
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
                pred_i.reid_feats = pred_boxes_t # trivial impl
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances
    def swin_backbone(self,x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()
        x=self.backbone(x)
        x=x[list(x.keys())[-1]]
        hw_shape=x.shape[2:]
        x=x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        bonenum=3
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum-1:
                norm_layer = getattr(self.swin, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        return {"stage3": outs[-1]}

@META_ARCH_REGISTRY.register()
class SwinF4RCNN2(SwinF4RCNN):
    @classmethod
    def from_config(cls, cfg):
        ret = SearchBase.from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin=SwinTransformer(semantic_weight=tr_cfg.SEMANTIC_WEIGHT,
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
            drop_path_rate=tr_cfg.DROP_PATH,)
        swin_out_shape={"stage{}".format(i+1): ShapeSpec(channels=swin.num_features[i],stride=swin.strides[i]) for i in range(len(swin.stages))}
        roi_heads=SwinROIHeads2(cfg,swin_out_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        ret.update({
            "proposal_generator": build_proposal_generator(
                cfg, swin_out_shape
            ),
            "roi_heads": roi_heads,
            "swin": swin,
        })
        return ret

    def swin_backbone(self,x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()
        x=self.backbone(x)
        x=x[list(x.keys())[-1]]
        hw_shape=x.shape[2:]
        x=x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        bonenum=3
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum-1:
                # norm_layer = getattr(self.swin, f'norm{i}')
                # out = norm_layer(out) NOTE this feat is to pass to the next stage, not the actual output
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        return {"stage3": outs[-1]}

@META_ARCH_REGISTRY.register()
class SwinF4RCNN3(SwinF4RCNN2):
    def swin_backbone(self,x): # keeps norm2
        return super(SwinPromptRCNN2,self).swin_backbone(x)

#NOTE prepend by default
@META_ARCH_REGISTRY.register()
class PromptedSwinF4RCNN(SwinF4RCNN):
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

        super(SwinF4RCNN,self).__init__(*args,**kws,)
        self.swin= swin
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.id_assigner=id_assigner
        for p in self.backbone.parameters():
            p.requires_grad=False
        for p in self.proposal_generator.parameters():
            p.requires_grad=False
        for p in self.roi_heads.parameters():
            p.requires_grad=False
        for n,p in self.swin.named_parameters():
            if "prompt" not in n : 
                p.requires_grad=False
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training=mode
        if mode:
            # training:
            for module in self.children():
                module.train(False)
            self.swin.prompt_dropout.train()
            self.roi_heads.training=True # for training vis
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = SearchBase.from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin=PromptedSwinTransformer(
            prompt_start_stage=tr_cfg.PROMPT_START_STAGE,# [1,2,3,4]
            num_prompts=tr_cfg.NUM_PROMPTS,
            prompt_drop_rate=tr_cfg.PROMPT_DROP_RATE,
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
            drop_path_rate=tr_cfg.DROP_PATH,)
        swin_out_shape={"stage{}".format(i+1): ShapeSpec(channels=swin.num_features[i],stride=swin.strides[i]) for i in range(len(swin.stages))}
        roi_heads=PromptedSwinROIHeads(cfg,swin_out_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        ret.update({
            "proposal_generator": build_proposal_generator(
                cfg, swin_out_shape
            ),
            "roi_heads": roi_heads,
            "swin": swin,
        })
        return ret
    def swin_backbone(self,x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()
        x=self.backbone(x)
        x=x[list(x.keys())[-1]]
        hw_shape=x.shape[2:]
        x=x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)
        if self.swin.promt_start_stage==0:
            x=self.swin.incorporate_prompt(x)

        outs = []
        bonenum=3
        for i, (stage,deep_prompt_embd) in enumerate(zip(self.swin.stages[:bonenum],[
                    self.swin.deep_prompt_embeddings_0,
                    self.swin.deep_prompt_embeddings_1,
                    self.swin.deep_prompt_embeddings_2,
                    self.swin.deep_prompt_embeddings_3
                ][:bonenum])):
            deep_prompt_embd = self.swin.prompt_dropout(deep_prompt_embd)
            if i<self.swin.promt_start_stage-1:
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape,deep_prompt_embd)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum-1:
                norm_layer = getattr(self.swin, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        return {"stage3": outs[-1]}

from psd2.modeling.roi_heads.roi_heads import ROIHeads,add_ground_truth_to_proposals
from psd2.modeling.poolers import ROIPooler
from psd2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from psd2.layers import ShapeSpec
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
    def _shared_roi_transform(self, features, boxes,swin):
        x = self.pooler(features, boxes)
        if swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()

        feat = x
        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum=3
        x,hw_shape = swin.stages[bonenum-1].downsample(x,hw_shape)
        if swin.semantic_weight >= 0:
            sw = swin.semantic_embed_w[bonenum-1](semantic_weight).unsqueeze(1)
            sb = swin.semantic_embed_b[bonenum-1](semantic_weight).unsqueeze(1)
            x = x * swin.softplus(sw) + sb
        for i, stage in enumerate(swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if swin.semantic_weight >= 0:
                sw = swin.semantic_embed_w[bonenum+i](semantic_weight).unsqueeze(1)
                sb = swin.semantic_embed_b[bonenum+i](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i == len(swin.stages) - bonenum - 1:
                norm_layer = getattr(swin, f'norm{bonenum+i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               swin.num_features[bonenum+i]).permute(0, 3, 1,
                                                             2).contiguous()
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
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
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
            proposals, pos_match_indices = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes,swin
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:

            losses = self.box_predictor.losses(predictions, proposals)
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

class SwinROIHeads2(SwinROIHeads):
    def _shared_roi_transform(self, features, boxes,swin):
        x = self.pooler(features, boxes)
        semantic_weight = 1.0 # this is only for detection, so give 1.0
        w = torch.ones(x.shape[0],1) * swin.semantic_weight
        w = torch.cat([w, 1-w], axis=-1)
        semantic_weight = w.cuda()

        feat = x
        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum=3
        x,hw_shape = swin.stages[bonenum-1].downsample(x,hw_shape)
        if swin.semantic_weight >= 0:
            sw = swin.semantic_embed_w[bonenum-1](semantic_weight).unsqueeze(1)
            sb = swin.semantic_embed_b[bonenum-1](semantic_weight).unsqueeze(1)
            x = x * swin.softplus(sw) + sb
        for i, stage in enumerate(swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if swin.semantic_weight >= 0:
                sw = swin.semantic_embed_w[bonenum+i](semantic_weight).unsqueeze(1)
                sb = swin.semantic_embed_b[bonenum+i](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i == len(swin.stages) - bonenum - 1:
                norm_layer = getattr(swin, f'norm{bonenum+i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               swin.num_features[bonenum+i]).permute(0, 3, 1,
                                                             2).contiguous()
        return out

class PromptedSwinROIHeads(SwinROIHeads):
    def _shared_roi_transform(self, features, boxes,swin):
        x = self.pooler(features, boxes) # no prompt tokens
        if swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()

        feat = x
        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum=3
        x,hw_shape = swin.stages[bonenum-1].downsample(x,hw_shape)
        if swin.semantic_weight >= 0:
            sw = swin.semantic_embed_w[bonenum-1](semantic_weight).unsqueeze(1)
            sb = swin.semantic_embed_b[bonenum-1](semantic_weight).unsqueeze(1)
            x = x * swin.softplus(sw) + sb
        # NOTE to simulate the added prompts in last stage
        trivial_prompt_emb =x.new_zeros((x.shape[0],swin.num_prompts,x.shape[-1]))
        x=torch.cat([trivial_prompt_emb,x],dim=1)
        for i, (stage,deep_prompt_embd) in enumerate(zip(swin.stages[bonenum:], [
                    swin.deep_prompt_embeddings_0,
                    swin.deep_prompt_embeddings_1,
                    swin.deep_prompt_embeddings_2,
                    swin.deep_prompt_embeddings_3
                ][bonenum:])):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape,deep_prompt_embd)
            x=x[:, swin.num_prompts:, :]
            if swin.semantic_weight >= 0:
                sw = swin.semantic_embed_w[bonenum+i](semantic_weight).unsqueeze(1)
                sb = swin.semantic_embed_b[bonenum+i](semantic_weight).unsqueeze(1)
                x = x * swin.softplus(sw) + sb
            if i == len(swin.stages) - bonenum - 1:
                norm_layer = getattr(swin, f'norm{bonenum+i}')
                # NOTE keeep the output tokens of prompts
                out = norm_layer(out) # B L C
                out = out.unsqueeze(1).permute(0, 3, 1,2).contiguous() # to be compatible with later average pooling
        return out

class FastRCNNOutputLayersPs(FastRCNNOutputLayers):
    def inference_unms(
        self, predictions, proposals
    ):
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


