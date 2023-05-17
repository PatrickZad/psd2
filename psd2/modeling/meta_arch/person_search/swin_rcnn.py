# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict, List, Optional
import torch
from torch import nn
import torch.nn.functional as tF
from psd2.config import configurable

from psd2.structures import ImageList, Instances, pairwise_iou, Boxes, BoxMode
from psd2.utils.events import get_event_storage
from psd2.layers import batched_nms
from .base import SearchBase

from ...proposal_generator import build_proposal_generator
from .. import META_ARCH_REGISTRY
from psd2.modeling.extend.solider import (
    SideSwinTransformer,
    SidePromptedSwinTransformer,
    SidePrefixPromptedSwinTransformer,
    SwinTransformer,
)
from psd2.modeling.id_assign import build_id_assigner
import psd2.modeling.prompts as prompts


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


from torch.nn import init
from psd2.layers.mem_matching_losses import OIMLoss
from torch.nn.functional import normalize


# NOTE to test finetune and upper-bound
@META_ARCH_REGISTRY.register()
class SwinF4RCNNPS(SwinF4RCNN):
    @configurable
    def __init__(
        self,
        reid_in_feature,
        oim_loss,
        bn_neck,
        pool_layer,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.reid_in_feature = reid_in_feature
        self.pool_layer = pool_layer
        self.oim_loss = oim_loss
        self.bn_neck = bn_neck
        for p in self.backbone.parameters():
            p.requires_grad = True
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
                p.requires_grad = True

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["reid_in_feature"] = cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["oim_loss"] = OIMLoss(cfg)
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        return ret

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

    def reid_head(self, image_list, backbone_features, roi_boxes, roi_ids=None):
        features = [backbone_features[self.reid_in_feature]]
        if self.training:
            roi_boxes = [
                Boxes(roi_boxes[i][roi_ids[i] > -2], box_mode=BoxMode.XYXY_ABS)
                for i in range(len(roi_ids))
            ]
            pos_ids = torch.cat(
                [roi_ids[i][roi_ids[i] > -2] for i in range(len(roi_ids))], dim=0
            )
        else:
            roi_boxes = [
                Boxes(roi_boxes_i, box_mode=BoxMode.XYXY_ABS)
                for roi_boxes_i in roi_boxes
            ]
        x = self.roi_heads.pooler(features, roi_boxes)
        del features
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum = 3
        x, hw_shape = self.swin.stages[bonenum - 1].downsample(x, hw_shape)
        if self.swin.semantic_weight >= 0:
            sw = self.swin.semantic_embed_w[bonenum - 1](semantic_weight).unsqueeze(1)
            sb = self.swin.semantic_embed_b[bonenum - 1](semantic_weight).unsqueeze(1)
            x = x * self.swin.softplus(sw) + sb
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[bonenum + i](semantic_weight).unsqueeze(
                    1
                )
                sb = self.swin.semantic_embed_b[bonenum + i](semantic_weight).unsqueeze(
                    1
                )
                x = x * self.swin.softplus(sw) + sb
            if i == len(self.swin.stages) - bonenum - 1:
                norm_layer = getattr(self.swin, f"norm{bonenum+i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[bonenum + i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        del x
        embs = self.pool_layer(out).reshape((out.shape[0], -1))
        embs = self.bn_neck(embs)
        if self.training:
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training_ps(
                        image_list.tensor,
                        out.split([len(bxs) for bxs in roi_boxes], dim=0),
                        [bxs.tensor for bxs in roi_boxes],
                    )
            del out
            reid_loss = self.oim_loss(embs, pos_ids)
            return reid_loss
        else:
            embs = normalize(embs, dim=-1)
            return torch.split(embs, [len(bxs) for bxs in roi_boxes])

    @torch.no_grad()
    def visualize_training_ps(self, images, p_feat_maps, p_boxes):
        storage = get_event_storage()
        trans_t2img_t = lambda t: t.detach().cpu() * self.pixel_std.cpu().view(
            -1, 1, 1
        ) + self.pixel_mean.cpu().view(-1, 1, 1)
        feat_map_size = p_feat_maps[0].shape[-2:]
        tg_size = [feat_map_size[0] * 16, feat_map_size[1] * 16]
        bs = len(p_boxes)
        for bi in range(bs):
            boxes_bi = p_boxes[bi].cpu()  # n x 4
            box_areas = (boxes_bi[:, 2] - boxes_bi[:, 0]) * boxes_bi[:, 3] - boxes_bi[
                :, 1
            ]
            sort_idxs = torch.argsort(box_areas, dim=0, descending=True)
            feats_bi = p_feat_maps[bi].cpu()  # n x roi_h x roi_w
            idxs = sort_idxs[: min(10, sort_idxs.shape[0])]
            img_rgb_t = trans_t2img_t(images[bi].cpu())  # 3 x h x w
            assigns_on_boxes = []
            for i in idxs:
                assign_on_box = _render_attn_on_box(
                    img_rgb_t * 255, boxes_bi[i], feats_bi[i], tgt_size=tg_size
                )
                assigns_on_boxes.append(assign_on_box)
            cat_assigns_on_boxes = torch.cat(assigns_on_boxes, dim=2)
            storage.put_image("img_{}/attn".format(bi), cat_assigns_on_boxes / 255.0)

    def forward_gallery(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.swin, image_list, features, proposals, gt_instances
        )
        roi_boxes = [inst.pred_boxes.tensor for inst in pred_instances]
        if self.training:
            losses.update(proposal_losses)
            assign_ids = self.id_assigner(
                roi_boxes,
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )
            reid_loss = self.reid_head(image_list, features, roi_boxes, assign_ids)
            losses.update(reid_loss)

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
            reid_feats = self.reid_head(image_list, features, roi_boxes)
            # nms
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i, feats_i, gt_i in zip(pred_instances, reid_feats, gt_instances):
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
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                pred_i.reid_feats = feats_i[keep]
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances

    def forward_query(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]


@META_ARCH_REGISTRY.register()
class SwinF4RCNNPS2(SwinF4RCNNPS):
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
        swin_out_norm_shape = {
            "stage{}_norm".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        swin_out_shape.update(swin_out_norm_shape)
        roi_heads = SwinROIHeads2(cfg, swin_out_shape)
        ret["roi_heads"] = roi_heads
        return ret

    def reid_head(self, image_list, backbone_features, roi_boxes, roi_ids=None):
        features = [backbone_features[self.reid_in_feature]]
        if self.training:
            roi_boxes = [
                Boxes(roi_boxes[i][roi_ids[i] > -2], box_mode=BoxMode.XYXY_ABS)
                for i in range(len(roi_ids))
            ]
            pos_ids = torch.cat(
                [roi_ids[i][roi_ids[i] > -2] for i in range(len(roi_ids))], dim=0
            )
        else:
            roi_boxes = [
                Boxes(roi_boxes_i, box_mode=BoxMode.XYXY_ABS)
                for roi_boxes_i in roi_boxes
            ]
        x = self.roi_heads.pooler(features, roi_boxes)
        del features
        # NOTE semantic_weight = 0:

        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum = 3
        x, hw_shape = self.swin.stages[bonenum - 1].downsample(x, hw_shape)
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i == len(self.swin.stages) - bonenum - 1:
                norm_layer = getattr(self.swin, f"norm{bonenum+i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[bonenum + i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        del x
        embs = self.pool_layer(out).reshape((out.shape[0], -1))
        embs = self.bn_neck(embs)
        if self.training:
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training_ps(
                        image_list.tensor,
                        out.split([len(bxs) for bxs in roi_boxes], dim=0),
                        [bxs.tensor for bxs in roi_boxes],
                    )
            del out
            reid_loss = self.oim_loss(embs, pos_ids)
            return reid_loss
        else:
            embs = normalize(embs, dim=-1)
            return torch.split(embs, [len(bxs) for bxs in roi_boxes])


from psd2.modeling.box_augmentation import build_box_augmentor
import copy


@META_ARCH_REGISTRY.register()
class SwinF4RCNNPS2BoxAug(SwinF4RCNNPS2):
    @configurable
    def __init__(self, box_aug, *args, **kws):
        super().__init__(*args, **kws)
        self.box_aug = box_aug

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["box_aug"] = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.swin, image_list, features, proposals, gt_instances
        )
        roi_boxes = [inst.pred_boxes.tensor for inst in pred_instances]
        if self.training:
            losses.update(proposal_losses)
            assign_ids = self.id_assigner(
                roi_boxes,
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )
            pos_boxes, pos_ids = [], []
            for gts_i in gt_instances:
                # append gt
                pos_boxes.append(gts_i.gt_boxes.tensor)
                pos_ids.append(gts_i.gt_pids)
            pos_boxes, pos_ids = self.box_aug.augment_boxes(
                pos_boxes,
                pos_ids,
                det_boxes=None,
                det_pids=None,
                img_sizes=[gti.image_size for gti in gt_instances],
            )
            reid_loss = self.reid_head(image_list, features, pos_boxes, pos_ids)
            losses.update(reid_loss)

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
            reid_feats = self.reid_head(image_list, features, roi_boxes)
            # nms
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i, feats_i, gt_i in zip(pred_instances, reid_feats, gt_instances):
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
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                pred_i.reid_feats = feats_i[keep]
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances


@META_ARCH_REGISTRY.register()
class PromptedSwinF4RCNNPS(SwinF4RCNNPS):
    @configurable
    def __init__(
        self,
        stage_prompts,
        side_stage_prompts,
        swin_org,
        swin_org_init_path,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.stage_prompts = stage_prompts
        self.side_stage_prompts = side_stage_prompts
        self.swin_org = swin_org
        self.swin_org_init_path = swin_org_init_path
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.swin.parameters():
            p.requires_grad = False
        for p in self.swin_org.parameters():
            p.requires_grad = False
        for p in self.proposal_generator.parameters():
            p.requires_grad = False
        for p in self.roi_heads.parameters():
            p.requires_grad = False

    def load_state_dict(self, *args, **kws):
        out = super().load_state_dict(*args, **kws)
        state_dict = _load_file(self.swin_org_init_path)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        org_ks = list(state_dict.keys())
        for k in org_ks:
            v = state_dict.pop(k)
            if k.startswith("swin."):
                nk = k[len("swin.") :]
                if not isinstance(v, torch.Tensor):
                    state_dict[nk] = torch.tensor(v, device=self.device)
        self.swin_org.load_state_dict(state_dict, strict=True)
        print("parameters of *swin_org* haved been loaded")
        return out

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.eval()
            self.swin.eval()
            self.swin_org.eval()
            self.proposal_generator.train()  # to train the early prompts via rpn loss
            self.roi_heads.train()  # to train the prompts via det loss
            self.stage_prompts.train()
            self.side_stage_prompts.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = super(SwinF4RCNNPS, cls).from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        if "L2P" in prompt_cfg.PROMPT_TYPE:
            num_prompts = prompt_cfg.NUM_PROMPTS * prompt_cfg.TOP_K
        else:
            num_prompts = prompt_cfg.NUM_PROMPTS
        # NOTE downsample module of stage3 is trainable
        swin = SidePromptedSwinTransformer(
            side_start_stage=3,
            prompt_start_stage=1,
            num_prompts=num_prompts,
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
        roi_heads = PromptedSwinROIHeads(cfg, swin_out_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        stage_prompts = nn.ModuleList()
        for si, nl in enumerate(tr_cfg.DEPTH):
            if prompt_cfg.PROMPT_TYPE == "L2P":
                prompt_stage = prompts.L2P(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PO":
                prompt_stage = prompts.L2POrg(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[si],
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.task_count = prompt_cfg.CURRECT_TASK
            stage_prompts.append(prompt_stage)
        side_stage_prompts = nn.ModuleList()
        if prompt_cfg.PROMPT_TYPE == "L2P":
            prompt_stage = prompts.L2P(
                emb_d=swin.num_features[-1],
                n_tasks=prompt_cfg.NUM_TASKS,
                pool_size=prompt_cfg.POOL_SIZE,
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
                topk=prompt_cfg.TOP_K,
                loss_weight=prompt_cfg.LOSS_WEIGHT,
                key_dim=swin.num_features[-1],
                vis_period=cfg.VIS_PERIOD,
            )
        elif prompt_cfg.PROMPT_TYPE == "L2PO":
            prompt_stage = prompts.L2POrg(
                emb_d=swin.num_features[-1],
                n_tasks=prompt_cfg.NUM_TASKS,
                pool_size=prompt_cfg.POOL_SIZE,
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
                topk=prompt_cfg.TOP_K,
                loss_weight=prompt_cfg.LOSS_WEIGHT,
                key_dim=swin.num_features[-1],
                vis_period=cfg.VIS_PERIOD,
            )
        elif prompt_cfg.PROMPT_TYPE == "Fixed":
            prompt_stage = prompts.FixedPrompts(
                emb_d=swin.num_features[-1],
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
            )
        else:  # CODAPrompt
            prompt_stage = prompts.CodaPrompt(
                emb_d=swin.num_features[-1],
                n_tasks=prompt_cfg.NUM_TASKS,
                pool_size=prompt_cfg.POOL_SIZE,
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
                loss_weight=prompt_cfg.LOSS_WEIGHT,
                key_dim=swin.num_features[-1],
                vis_period=cfg.VIS_PERIOD,
            )
        prompt_stage.task_count = prompt_cfg.CURRECT_TASK
        side_stage_prompts.append(prompt_stage)

        ret.update(
            {
                "proposal_generator": build_proposal_generator(cfg, swin_out_shape),
                "roi_heads": roi_heads,
                "swin": swin,
                "stage_prompts": stage_prompts,
                "side_stage_prompts": side_stage_prompts,
            }
        )
        swin_org = SwinTransformer(
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
        )
        ret["swin_org"] = swin_org
        ret["swin_org_init_path"] = cfg.PERSON_SEARCH.QUERY_ENCODER_WEIGHTS
        ret["reid_in_feature"] = cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["oim_loss"] = OIMLoss(cfg)
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        return ret

    @torch.no_grad()
    def task_query(self, backbone_features):
        x = backbone_features[list(backbone_features.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin_org.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin_org.drop_after_pos(x)

        if self.swin_org.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin_org.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        for i, stage in enumerate(self.swin_org.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin_org.semantic_weight >= 0:
                sw = self.swin_org.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin_org.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin_org.softplus(sw) + sb
        out = self.swin_org.norm3(out)
        out = (
            out.view(-1, *out_hw_shape, self.swin_org.num_features[i])
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x = self.swin_org.avgpool(out)
        x = torch.flatten(x, 1)
        return x

    def swin_backbone(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()
        x = self.backbone(x)
        task_query = self.task_query(x)

        x = x[list(x.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        prompt_loss = 0
        bonenum = 3
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            task_query_stage = task_query_x.expand(-1, len(stage.blocks), -1)
            selected_prompts, p_loss = self.stage_prompts[i](
                task_query_stage, i, train=self.training
            )
            prompt_loss += p_loss
            x, hw_shape, out, out_hw_shape = stage(
                x, hw_shape, deep_prompt_embd=selected_prompts
            )
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
        if self.training:
            prompt_loss = {"loss_prompt": prompt_loss}
        return {"stage3": outs[-1]}, task_query, prompt_loss

    def reid_head(
        self, task_query, image_list, backbone_features, roi_boxes, roi_ids=None
    ):
        features = [backbone_features[self.reid_in_feature]]
        if self.training:
            roi_boxes = [
                Boxes(roi_boxes[i][roi_ids[i] > -2], box_mode=BoxMode.XYXY_ABS)
                for i in range(len(roi_ids))
            ]
            pos_ids = torch.cat(
                [roi_ids[i][roi_ids[i] > -2] for i in range(len(roi_ids))], dim=0
            )
        else:
            roi_boxes = [
                Boxes(roi_boxes_i, box_mode=BoxMode.XYXY_ABS)
                for roi_boxes_i in roi_boxes
            ]
        x = self.roi_heads.pooler(features, roi_boxes)
        del features
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        bonenum = 3
        x, hw_shape = self.swin.stages[bonenum - 1].downsample(x, hw_shape)
        if self.swin.semantic_weight >= 0:
            sw = self.swin.semantic_embed_w[bonenum - 1](semantic_weight).unsqueeze(1)
            sb = self.swin.semantic_embed_b[bonenum - 1](semantic_weight).unsqueeze(1)
            x = x * self.swin.softplus(sw) + sb
        prompt_loss = 0
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            task_query_stage = task_query_x.expand(-1, len(stage.blocks), -1)
            selected_prompts, p_loss = self.stage_prompts[bonenum + i](
                task_query_stage, f"reid{i}", train=self.training
            )
            prompt_loss += p_loss
            expanded_prompts = []
            for bi in range(len(roi_boxes)):
                expanded_prompts.append(
                    selected_prompts[:, bi : bi + 1].expand(
                        -1, len(roi_boxes[bi]), -1, -1
                    )
                )
            selected_prompts = torch.cat(expanded_prompts, dim=1)
            del expanded_prompts
            x, hw_shape, out, out_hw_shape = stage(
                x, hw_shape, deep_prompt_embd=selected_prompts
            )
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[bonenum + i](semantic_weight).unsqueeze(
                    1
                )
                sb = self.swin.semantic_embed_b[bonenum + i](semantic_weight).unsqueeze(
                    1
                )
                x = x * self.swin.softplus(sw) + sb
            if i == len(self.swin.stages) - bonenum - 1:
                norm_layer = getattr(self.swin, f"norm{bonenum+i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[bonenum + i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        del x
        embs = self.pool_layer(out).reshape((out.shape[0], -1))
        embs = self.bn_neck(embs)
        if self.training:
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training_ps(
                        image_list.tensor,
                        out.split([len(bxs) for bxs in roi_boxes], dim=0),
                        [bxs.tensor for bxs in roi_boxes],
                    )
            del out
            reid_loss = self.oim_loss(embs, pos_ids)
            reid_loss.update({"loss_prompt_reid": prompt_loss})
            return reid_loss
        else:
            embs = normalize(embs, dim=-1)
            return torch.split(embs, [len(bxs) for bxs in roi_boxes])

    def forward_gallery(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.swin,
            self.side_stage_prompts,
            task_query,
            image_list,
            features,
            proposals,
            gt_instances,
        )
        roi_boxes = [inst.pred_boxes.tensor for inst in pred_instances]
        if self.training:
            losses.update(proposal_losses)
            assign_ids = self.id_assigner(
                roi_boxes,
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )
            reid_loss = self.reid_head(
                task_query, image_list, features, roi_boxes, assign_ids
            )
            losses.update(reid_loss)
            losses.update(prompt_loss)

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
            reid_feats = self.reid_head(task_query, image_list, features, roi_boxes)
            # nms
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i, feats_i, gt_i in zip(pred_instances, reid_feats, gt_instances):
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
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                pred_i.reid_feats = feats_i[keep]
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances

    def forward_query(self, image_list, gt_instances):
        features, task_query, _ = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(task_query, image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]


@META_ARCH_REGISTRY.register()
class PrefixPromptedSwinF4RCNNPS(PromptedSwinF4RCNNPS):
    @classmethod
    def from_config(cls, cfg):
        ret = super(SwinF4RCNNPS, cls).from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        if "L2P" in prompt_cfg.PROMPT_TYPE:
            num_prompts = prompt_cfg.NUM_PROMPTS * prompt_cfg.TOP_K
        else:
            num_prompts = prompt_cfg.NUM_PROMPTS
        # NOTE downsample module of stage3 is trainable
        swin = SidePrefixPromptedSwinTransformer(
            side_start_stage=3,
            prompt_start_stage=1,
            num_prompts=num_prompts,
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
        roi_heads = PromptedSwinROIHeads(cfg, swin_out_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        stage_prompts = nn.ModuleList()
        for si, nl in enumerate(tr_cfg.DEPTH):
            if prompt_cfg.PROMPT_TYPE == "L2P":
                prompt_stage = prompts.L2P(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PO":
                prompt_stage = prompts.L2POrg(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[si],
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.task_count = prompt_cfg.CURRECT_TASK
            stage_prompts.append(prompt_stage)
        side_stage_prompts = nn.ModuleList()
        if prompt_cfg.PROMPT_TYPE == "L2P":
            prompt_stage = prompts.L2P(
                emb_d=swin.num_features[-1],
                n_tasks=prompt_cfg.NUM_TASKS,
                pool_size=prompt_cfg.POOL_SIZE,
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
                topk=prompt_cfg.TOP_K,
                loss_weight=prompt_cfg.LOSS_WEIGHT,
                key_dim=swin.num_features[-1],
                vis_period=cfg.VIS_PERIOD,
            )
        elif prompt_cfg.PROMPT_TYPE == "L2PO":
            prompt_stage = prompts.L2POrg(
                emb_d=swin.num_features[-1],
                n_tasks=prompt_cfg.NUM_TASKS,
                pool_size=prompt_cfg.POOL_SIZE,
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
                topk=prompt_cfg.TOP_K,
                loss_weight=prompt_cfg.LOSS_WEIGHT,
                key_dim=swin.num_features[-1],
                vis_period=cfg.VIS_PERIOD,
            )
        elif prompt_cfg.PROMPT_TYPE == "Fixed":
            prompt_stage = prompts.FixedPrompts(
                emb_d=swin.num_features[-1],
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
            )
        else:  # CODAPrompt
            prompt_stage = prompts.CodaPrompt(
                emb_d=swin.num_features[-1],
                n_tasks=prompt_cfg.NUM_TASKS,
                pool_size=prompt_cfg.POOL_SIZE,
                num_prompts=prompt_cfg.NUM_PROMPTS,
                num_layers=tr_cfg.DEPTH[-1],
                loss_weight=prompt_cfg.LOSS_WEIGHT,
                key_dim=swin.num_features[-1],
                vis_period=cfg.VIS_PERIOD,
            )
        prompt_stage.task_count = prompt_cfg.CURRECT_TASK
        side_stage_prompts.append(prompt_stage)

        ret.update(
            {
                "proposal_generator": build_proposal_generator(cfg, swin_out_shape),
                "roi_heads": roi_heads,
                "swin": swin,
                "stage_prompts": stage_prompts,
                "side_stage_prompts": side_stage_prompts,
            }
        )
        swin_org = SwinTransformer(
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
        )
        ret["swin_org"] = swin_org
        ret["swin_org_init_path"] = cfg.PERSON_SEARCH.QUERY_ENCODER_WEIGHTS
        ret["reid_in_feature"] = cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["oim_loss"] = OIMLoss(cfg)
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        return ret


from psd2.modeling.roi_heads.roi_heads import ROIHeads, add_ground_truth_to_proposals
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
        prompt_loss = 0
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(
            swin.side_stages[bonenum - swin.side_start_stage + 1 :]
        ):
            task_query_stage = task_query_x.expand(-1, len(stage.blocks), -1)
            selected_prompts, p_loss = det_stage_prompts[i](
                task_query_stage, f"det{i}", train=self.training
            )
            prompt_loss += p_loss
            expanded_prompts = []
            for bi in range(len(boxes)):
                expanded_prompts.append(
                    selected_prompts[:, bi : bi + 1].expand(-1, len(boxes[bi]), -1, -1)
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


import cv2


def _render_attn_on_box(img_rgb_t, pbox_t, feat_box, tgt_size=(384, 192)):
    # NOTE split two view for clearity
    pbox_int = pbox_t.int()  # xyxy
    pbox_img_rgb_t = img_rgb_t[
        :,
        max(pbox_int[1], 0) : min(pbox_int[3] + 1, img_rgb_t.shape[1]),
        max(pbox_int[0], 0) : min(pbox_int[2] + 1, img_rgb_t.shape[2]),
    ].clone()  # 3 x h x w
    pbox_img_rgb_t = torch.nn.functional.interpolate(
        pbox_img_rgb_t[None], size=tgt_size, mode="bilinear", align_corners=False
    ).squeeze(0)

    attn_reshaped = torch.nn.functional.interpolate(
        feat_box[None], size=tgt_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    attn_reshaped = (attn_reshaped**2).sum(0)
    attn_reshaped = (
        255
        * (attn_reshaped - attn_reshaped.min())
        / (attn_reshaped.max() - attn_reshaped.min() + 1e-12)
    )
    attn_img = attn_reshaped.cpu().numpy().astype(np.uint8)
    attn_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_JET)  # bgr hwc
    attn_img = torch.tensor(
        attn_img[..., ::-1].copy(), device=img_rgb_t.device, dtype=img_rgb_t.dtype
    ).permute(2, 0, 1)
    coeff = 0.3
    pbox_img_rgb_t = (1 - coeff) * pbox_img_rgb_t + coeff * attn_img

    return pbox_img_rgb_t


def _load_file(filename):
    from psd2.utils.file_io import PathManager
    import pickle

    if filename.endswith(".pkl"):
        with PathManager.open(filename, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        if "model" in data and "__author__" in data:
            # file is in Detectron2 model zoo format
            return data
        else:
            # assume file is from Caffe2 / Detectron1 model zoo
            if "blobs" in data:
                # Detection models have "blobs", but ImageNet models don't
                data = data["blobs"]
            data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
            return {
                "model": data,
                "__author__": "Caffe2",
                "matching_heuristics": True,
            }
    elif filename.endswith(".pyth"):
        # assume file is from pycls; no one else seems to use the ".pyth" extension
        with PathManager.open(filename, "rb") as f:
            data = torch.load(f)
        assert (
            "model_state" in data
        ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
        model_state = {
            k: v
            for k, v in data["model_state"].items()
            if not k.endswith("num_batches_tracked")
        }
        return {
            "model": model_state,
            "__author__": "pycls",
            "matching_heuristics": True,
        }

    loaded = torch.load(
        filename, map_location=torch.device("cpu")
    )  # load native pth checkpoint
    if "model" not in loaded:
        loaded = {"model": loaded}
    return loaded
