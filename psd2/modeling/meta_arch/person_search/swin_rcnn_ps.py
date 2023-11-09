# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np

import torch
from torch import nn
from psd2.config import configurable

from psd2.structures import Instances, Boxes, BoxMode
from psd2.utils.events import get_event_storage
from psd2.layers import batched_nms


from ...proposal_generator import build_proposal_generator
from .. import META_ARCH_REGISTRY
from psd2.modeling.extend.solider import (
    SidePromptedSwinTransformer,
    SidePrefixPromptedSwinTransformer,
    SwinTransformer,
    PatchMerging
)
from psd2.modeling.id_assign import build_id_assigner
import psd2.modeling.prompts as prompts

from torch.nn import init
from psd2.layers.mem_matching_losses import OIMLoss, IncOIMLoss
from torch.nn.functional import normalize
from .swin_rcnn_pd import (
    SwinF4RCNN,
    SwinF4RCNNSeq,
    SwinROIHeads2,
    PromptedSwinROIHeads,
    SwinSimFPNRCNN,
    SimpleFeaturePyramid,
    AlteredStandaredROIHeads,
    AlteredStandaredROIHeadsTi,
    PromptedSeqSwinROIHeads,
)
from psd2.modeling import ShapeSpec
from psd2.modeling.poolers import ROIPooler


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
        reid_box_pooler,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.reid_in_feature = reid_in_feature
        self.pool_layer = pool_layer
        self.oim_loss = oim_loss
        self.bn_neck = bn_neck
        self.reid_box_pooler = reid_box_pooler
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
        ret["oim_loss"] = (
            OIMLoss(cfg)
            if hasattr(cfg.PERSON_SEARCH.REID.LOSS, "OIM")
            else IncOIMLoss(cfg)
        )
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=ret["swin"].num_features[i], stride=ret["swin"].strides[i]
            )
            for i in range(len(ret["swin"].stages))
        }
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if isinstance(pooler_resolution, int):
            pooler_resolution = (pooler_resolution, pooler_resolution)
        assert len(pooler_resolution) == 2
        assert isinstance(pooler_resolution[0], int) and isinstance(
            pooler_resolution[1], int
        )
        pooler_scales = tuple(
            1.0 / swin_out_shape[k].stride
            for k in [cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT]
        )
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        reid_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["reid_box_pooler"] = reid_box_pooler
        return ret

    def load_state_dict(self, *args, **kws):
        state_dict = args[0]
        if (
            isinstance(self.oim_loss, IncOIMLoss)
            and "oim_loss.lb_layers.0.lookup_table" not in state_dict
        ):
            # resume from plain oim trained
            if "oim_loss.lb_layer.lookup_table" in state_dict:
                lkt0 = state_dict.pop("oim_loss.lb_layer.lookup_table")
                state_dict["oim_loss.lb_layers.0.lookup_table"] = lkt0
            if "oim_loss.ulb_layer.queue" in state_dict:
                q0 = state_dict.pop("oim_loss.ulb_layer.queue")
                state_dict["oim_loss.ulb_layers.0.queue"] = q0
            if "oim_loss.ulb_layer.tail" in state_dict:
                q0 = state_dict.pop("oim_loss.ulb_layer.tail")
                state_dict["oim_loss.ulb_layers.0.tail"] = q0
        return super().load_state_dict(*args, **kws)

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
        x = self.reid_box_pooler(features, roi_boxes)
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

    def forward_gallery_vis(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.swin, image_list, features, proposals, gt_instances
        )
        roi_boxes = [inst.pred_boxes.tensor for inst in pred_instances]
        # nms
        reid_feats = self.reid_head(image_list, features, roi_boxes)
        score_t = self.roi_heads.box_predictor.test_score_thresh
        iou_t = self.roi_heads.box_predictor.test_nms_thresh
        for pred_i, feats_i in zip(pred_instances, reid_feats):
            pred_boxes_t = pred_i.pred_boxes.tensor
            pred_scores = pred_i.pred_scores
            filter_mask = pred_scores >= score_t
            pred_boxes_t = pred_boxes_t[filter_mask]
            pred_scores = pred_scores[filter_mask]
            cate_idx = pred_scores.new_zeros(pred_scores.shape[0], dtype=torch.int64)
            # nms
            keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)
            pred_boxes_t = pred_boxes_t[keep]
            pred_scores = pred_scores[keep]
            pred_i.pred_boxes = Boxes(pred_boxes_t, BoxMode.XYXY_ABS)
            pred_i.pred_scores = pred_scores
            pred_i.reid_feats = feats_i[keep]
            pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
        return pred_instances, [feat.detach() for feat in features.values()]

    def forward_gallery_gt(self, image_list, gt_instances):
        assert not self.training
        features = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        reid_feats = self.reid_head(image_list, features, roi_boxes)
        pred_instances = []
        for i, (gt_i, feats_i) in enumerate(zip(gt_instances, reid_feats)):
            inst = Instances(gt_i.image_size)
            inst.pred_boxes = gt_i.gt_boxes
            inst.pred_scores = (
                torch.zeros_like(gt_i.gt_pids, dtype=feats_i.dtype) + 0.99
            )
            inst.pred_classes = torch.zeros_like(gt_i.gt_pids)
            inst.assign_ids = gt_i.gt_pids
            inst.reid_feats = feats_i
            # back to org scale
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            inst.pred_boxes.scale(org_w / w, org_h / h)
            pred_instances.append(inst)

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
class SwinSimFPNRCNNPS(SwinSimFPNRCNN):
    @configurable
    def __init__(
        self,
        reid_in_feature,
        oim_loss,
        bn_neck,
        pool_layer,
        reid_box_pooler,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.reid_in_feature = reid_in_feature
        self.pool_layer = pool_layer
        self.oim_loss = oim_loss
        self.bn_neck = bn_neck
        self.reid_box_pooler = reid_box_pooler
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
        ret["oim_loss"] = (
            OIMLoss(cfg)
            if hasattr(cfg.PERSON_SEARCH.REID.LOSS, "OIM")
            else IncOIMLoss(cfg)
        )
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=ret["swin"].num_features[i], stride=ret["swin"].strides[i]
            )
            for i in range(len(ret["swin"].stages))
        }
        swin_out_shape.update(
            {
                "side_stage{}".format(i + 1): ShapeSpec(
                    channels=ret["swin"].num_features[i], stride=ret["swin"].strides[i]
                )
                for i in range(len(ret["swin"].stages))
            }
        )
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if isinstance(pooler_resolution, int):
            pooler_resolution = (pooler_resolution, pooler_resolution)
        assert len(pooler_resolution) == 2
        assert isinstance(pooler_resolution[0], int) and isinstance(
            pooler_resolution[1], int
        )
        pooler_scales = tuple(
            1.0 / swin_out_shape[k].stride
            for k in [cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT]
        )
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        reid_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["reid_box_pooler"] = reid_box_pooler
        return ret

    def load_state_dict(self, *args, **kws):
        state_dict = args[0]
        if (
            isinstance(self.oim_loss, IncOIMLoss)
            and "oim_loss.lb_layers.0.lookup_table" not in state_dict
        ):
            # resume from plain oim trained
            if "oim_loss.lb_layer.lookup_table" in state_dict:
                lkt0 = state_dict.pop("oim_loss.lb_layer.lookup_table")
                state_dict["oim_loss.lb_layers.0.lookup_table"] = lkt0
            if "oim_loss.ulb_layer.queue" in state_dict:
                q0 = state_dict.pop("oim_loss.ulb_layer.queue")
                state_dict["oim_loss.ulb_layers.0.queue"] = q0
            if "oim_loss.ulb_layer.tail" in state_dict:
                q0 = state_dict.pop("oim_loss.ulb_layer.tail")
                state_dict["oim_loss.ulb_layers.0.tail"] = q0
        return super().load_state_dict(*args, **kws)

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
        x = self.reid_box_pooler(features, roi_boxes)
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
        fpn_features = self.sim_fpn(features)
        proposals, proposal_losses = self.proposal_generator(
            image_list, fpn_features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            image_list, fpn_features, proposals, gt_instances
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

    def forward_gallery_gt(self, image_list, gt_instances):
        assert not self.training
        features = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        reid_feats = self.reid_head(image_list, features, roi_boxes)
        pred_instances = []
        for i, (gt_i, feats_i) in enumerate(zip(gt_instances, reid_feats)):
            inst = Instances(gt_i.image_size)
            inst.pred_boxes = gt_i.gt_boxes
            inst.pred_scores = (
                torch.zeros_like(gt_i.gt_pids, dtype=feats_i.dtype) + 0.99
            )
            inst.pred_classes = torch.zeros_like(gt_i.gt_pids)
            inst.assign_ids = gt_i.gt_pids
            inst.reid_feats = feats_i
            # back to org scale
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            inst.pred_boxes.scale(org_w / w, org_h / h)
            pred_instances.append(inst)

        return pred_instances

    def forward_query(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]

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
        rst = {}
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i >= bonenum - 2:
                norm_layer = getattr(self.swin, f"side_norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return {"stage3": outs[0], "stage4": outs[1]}


@META_ARCH_REGISTRY.register()
class SwinF4RCNNSeqPS(SwinF4RCNNSeq):
    @configurable
    def __init__(
        self,
        oim_loss,
        bn_neck,
        pool_layer,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
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
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["oim_loss"] = (
            OIMLoss(cfg)
            if hasattr(cfg.PERSON_SEARCH.REID.LOSS, "OIM")
            else IncOIMLoss(cfg)
        )
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        return ret

    def load_state_dict(self, *args, **kws):
        state_dict = args[0]
        if (
            isinstance(self.oim_loss, IncOIMLoss)
            and "oim_loss.lb_layers.0.lookup_table" not in state_dict
        ):
            # resume from plain oim trained
            if "oim_loss.lb_layer.lookup_table" in state_dict:
                lkt0 = state_dict.pop("oim_loss.lb_layer.lookup_table")
                state_dict["oim_loss.lb_layers.0.lookup_table"] = lkt0
            if "oim_loss.ulb_layer.queue" in state_dict:
                q0 = state_dict.pop("oim_loss.ulb_layer.queue")
                state_dict["oim_loss.ulb_layers.0.queue"] = q0
            if "oim_loss.ulb_layer.tail" in state_dict:
                q0 = state_dict.pop("oim_loss.ulb_layer.tail")
                state_dict["oim_loss.ulb_layers.0.tail"] = q0
        return super().load_state_dict(*args, **kws)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

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

    def reid_head(self,image_list,box_features,roi_boxes,box_ids=None):
        embs = self.pool_layer(box_features).reshape((box_features.shape[0], -1))
        embs = self.bn_neck(embs)
        if self.training:
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training_ps(
                        image_list.tensor,
                        box_features.split([len(bxs) for bxs in roi_boxes], dim=0),
                        roi_boxes,
                    )
            del box_features
            reid_loss = self.oim_loss(embs, box_ids)
            return reid_loss
        else:
            embs = normalize(embs, dim=-1)
            return torch.split(embs, [len(bxs) for bxs in roi_boxes])

    def forward_gallery(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices,box_features = self.roi_heads(
            image_list, features, proposals, gt_instances,return_box_feat=True
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
            reid_loss = self.reid_head(image_list, box_features, roi_boxes, assign_ids)
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
            reid_feats = self.reid_head(image_list, box_features, roi_boxes)
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

    def forward_gallery_vis(self, image_list, gt_instances):
        raise NotImplementedError

    def forward_gallery_gt(self, image_list, gt_instances):
        raise NotImplementedError

    def forward_query(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_features=self.roi_heads.forward_gt(
            features,gt_instances
        )
        box_embs = self.reid_head(image_list, box_features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]

import copy
@META_ARCH_REGISTRY.register()
class PromptedSwinF4RCNNSeqPS(SwinF4RCNNSeqPS):
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
        # NOTE roi_heads manager its own parameters
        #for p in self.roi_heads.parameters():
        #    p.requires_grad = False
        for p in self.side_stage_prompts.parameters():
            p.requires_grad = False
        for p in self.stage_prompts[3:].parameters():
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
        res = self.swin_org.load_state_dict(state_dict, strict=False)
        print("parameters of *swin_org* haved been loaded: \n")
        print(res)
        return out
    @classmethod
    def from_config(cls, cfg):
        ret = super(SwinF4RCNNSeqPS, cls).from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 4
        in_num_prompts = (
            [n * prompt_cfg.TOP_K for n in num_prompts]
            if "L2P" in prompt_cfg.PROMPT_TYPE and "Attn" not in prompt_cfg.PROMPT_TYPE
            else num_prompts
        )
        # NOTE downsample module of stage3 is trainable
        swin = SidePromptedSwinTransformer(
            side_start_stage=3,
            prompt_start_stage=1,
            num_prompts=in_num_prompts,
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
            strides=(4, 2, 2, 1),
        )
        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        
        ret["id_assigner"] = build_id_assigner(cfg)
        stage_prompts = nn.ModuleList()
        for si, nl in enumerate(tr_cfg.DEPTH):
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            if stage_num_prompts == 0:
                stage_prompts.append(nn.Identity())
                continue
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[si],
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
            stage_prompts.append(prompt_stage)
        side_stage_prompts = nn.ModuleList()
        if isinstance(num_prompts, int):
            stage_num_prompts = num_prompts
        else:
            stage_num_prompts = num_prompts[-1]
        if stage_num_prompts == 0:
            stage_prompts.append(nn.Identity())
        else:
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[-1],
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
            side_stage_prompts.append(prompt_stage)

        head_swin=copy.deepcopy(swin)
        org_ds = head_swin.side_stages[-2].downsample
        # NOTE dirty impl
        head_swin.side_stages[-2].downsample = PatchMerging(
            in_channels=org_ds.in_channels,
            out_channels=org_ds.out_channels,
            stride=2,
            norm_cfg=dict(type="LN"),
            init_cfg=None,
        )
        roi_heads = PromptedSeqSwinROIHeads(cfg, swin_out_shape,head_swin,copy.deepcopy(stage_prompts),copy.deepcopy(side_stage_prompts))
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
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["oim_loss"] = (
            OIMLoss(cfg)
            if hasattr(cfg.PERSON_SEARCH.REID.LOSS, "OIM")
            else IncOIMLoss(cfg)
        )
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))

        return ret

    def train(self, mode=True):
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
        prompt_loss = torch.tensor(
            0., dtype=task_query.dtype, device=task_query.device
        )
        bonenum = 3
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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
    def forward_gallery(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices,box_features = self.roi_heads(
            task_query,image_list, features, proposals, gt_instances,return_box_feat=True
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
            reid_loss = self.reid_head(image_list, box_features, roi_boxes, torch.cat(assign_ids,dim=0))
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
            reid_feats = self.reid_head(image_list, box_features, roi_boxes)
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
        box_features=self.roi_heads.forward_gt(
            task_query,features,gt_instances
        )
        box_embs = self.reid_head(image_list, box_features, roi_boxes)
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
        x = self.reid_box_pooler(features, roi_boxes)
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
class SwinSimFPNLiteRCNNPS2BoxAugOpen(SwinSimFPNRCNNPS):
    @configurable
    def __init__(self, box_aug, *args, **kws):
        super().__init__(*args, **kws)
        self.box_aug = box_aug
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
        ret["box_aug"] = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
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
                norm_layer = getattr(self.swin, f"side_norm{i}")
                outd = norm_layer(out)
                outd = (
                    outd.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(outd)  # for det
                norm_layer = getattr(self.swin, f"norm{i}")
                outr = norm_layer(out)
                outr = (
                    outr.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(outr)  # for det
        return {"side_stage3": outs[0], "stage3": outs[1]}

    def forward_gallery(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        fpn_features = self.sim_fpn(features)
        proposals, proposal_losses = self.proposal_generator(
            image_list, fpn_features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            image_list, fpn_features, proposals, gt_instances
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
        res = self.swin_org.load_state_dict(state_dict, strict=False)
        print("parameters of *swin_org* haved been loaded: \n")
        print(res)
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
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 4
        in_num_prompts = (
            [n * prompt_cfg.TOP_K for n in num_prompts]
            if "L2P" in prompt_cfg.PROMPT_TYPE and "Attn" not in prompt_cfg.PROMPT_TYPE
            else num_prompts
        )
        # NOTE downsample module of stage3 is trainable
        swin = SidePromptedSwinTransformer(
            side_start_stage=3,
            prompt_start_stage=1,
            num_prompts=in_num_prompts,
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
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            if stage_num_prompts == 0:
                stage_prompts.append(nn.Identity())
                continue
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[si],
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
            stage_prompts.append(prompt_stage)
        side_stage_prompts = nn.ModuleList()
        if isinstance(num_prompts, int):
            stage_num_prompts = num_prompts
        else:
            stage_num_prompts = num_prompts[-1]
        if stage_num_prompts == 0:
            stage_prompts.append(nn.Identity())
        else:
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[-1],
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
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
        ret["oim_loss"] = (
            OIMLoss(cfg)
            if hasattr(cfg.PERSON_SEARCH.REID.LOSS, "OIM")
            else IncOIMLoss(cfg)
        )
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if isinstance(pooler_resolution, int):
            pooler_resolution = (pooler_resolution, pooler_resolution)
        assert len(pooler_resolution) == 2
        assert isinstance(pooler_resolution[0], int) and isinstance(
            pooler_resolution[1], int
        )
        pooler_scales = tuple(
            1.0 / swin_out_shape[k].stride
            for k in [cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT]
        )
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        reid_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["reid_box_pooler"] = reid_box_pooler
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
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        bonenum = 3
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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
        x = self.reid_box_pooler(features, roi_boxes)
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
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i + bonenum] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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

    def forward_gallery_vis(self, image_list, gt_instances):
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
            cate_idx = pred_scores.new_zeros(pred_scores.shape[0], dtype=torch.int64)
            # nms
            keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)
            pred_boxes_t = pred_boxes_t[keep]
            pred_scores = pred_scores[keep]
            pred_i.pred_boxes = Boxes(pred_boxes_t, BoxMode.XYXY_ABS)
            pred_i.pred_scores = pred_scores
            pred_i.reid_feats = pred_boxes_t  # trivial impl
            pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
            pred_i.assign_ids = pred_i.assign_ids[filter_mask][keep]
        return pred_instances, [feat.detach() for feat in features.values()]

    def forward_gallery_gt(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        reid_feats = self.reid_head(task_query, image_list, features, roi_boxes)
        pred_instances = []
        for i, (gt_i, feats_i) in enumerate(zip(gt_instances, reid_feats)):
            inst = Instances(gt_i.image_size)
            inst.pred_boxes = gt_i.gt_boxes
            inst.pred_scores = (
                torch.zeros_like(gt_i.gt_pids, dtype=feats_i.dtype) + 0.99
            )
            inst.pred_classes = torch.zeros_like(gt_i.gt_pids)
            inst.assign_ids = gt_i.gt_pids
            inst.reid_feats = feats_i
            # back to org scale
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            inst.pred_boxes.scale(org_w / w, org_h / h)
            pred_instances.append(inst)

        return pred_instances

    def forward_query(self, image_list, gt_instances):
        features, task_query, _ = self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(task_query, image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]


from psd2.modeling.backbone.fpn import LastLevelMaxPool


@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNRCNNPS(SwinSimFPNRCNNPS):
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
        for p in self.sim_fpn.parameters():
            p.requires_grad = False

        self.pred_rst = None

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
        res = self.swin_org.load_state_dict(state_dict, strict=False)
        print("parameters of *swin_org* haved been loaded: \n")
        print(res)
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
            self.sim_fpn.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = super(SwinSimFPNRCNNPS, cls).from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        num_prompts = prompt_cfg.NUM_PROMPTS
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 4
        in_num_prompts = (
            [n * prompt_cfg.TOP_K for n in num_prompts]
            if "L2P" in prompt_cfg.PROMPT_TYPE and "Attn" not in prompt_cfg.PROMPT_TYPE
            else num_prompts
        )
        # NOTE downsample module of stage3 is trainable
        swin = SidePromptedSwinTransformer(
            side_start_stage=3,
            prompt_start_stage=1,
            num_prompts=in_num_prompts,
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
        swin_out_shape.update(
            {
                "side_stage{}".format(i + 1): ShapeSpec(
                    channels=swin.num_features[i], stride=swin.strides[i]
                )
                for i in range(len(swin.stages))
            }
        )
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
        stage_prompts = nn.ModuleList()
        for si, nl in enumerate(tr_cfg.DEPTH):
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            if stage_num_prompts == 0:
                stage_prompts.append(nn.Identity())
                continue
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[si],
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
            stage_prompts.append(prompt_stage)
        side_stage_prompts = nn.ModuleList()
        if isinstance(num_prompts, int):
            stage_num_prompts = num_prompts
        else:
            stage_num_prompts = num_prompts[-1]
        if stage_num_prompts == 0:
            stage_prompts.append(nn.Identity())
        else:
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[-1],
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
            side_stage_prompts.append(prompt_stage)

        ret.update(
            {
                "proposal_generator": build_proposal_generator(
                    cfg, sim_fpn.output_shape()
                ),
                "roi_heads": roi_heads,
                "swin": swin,
                "sim_fpn": sim_fpn,
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
        ret["oim_loss"] = (
            OIMLoss(cfg)
            if hasattr(cfg.PERSON_SEARCH.REID.LOSS, "OIM")
            else IncOIMLoss(cfg)
        )
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if isinstance(pooler_resolution, int):
            pooler_resolution = (pooler_resolution, pooler_resolution)
        assert len(pooler_resolution) == 2
        assert isinstance(pooler_resolution[0], int) and isinstance(
            pooler_resolution[1], int
        )
        pooler_scales = tuple(
            1.0 / swin_out_shape[k].stride
            for k in [cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT]
        )
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        reid_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["reid_box_pooler"] = reid_box_pooler
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
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        bonenum = 4
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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
            if i >= bonenum - 2:
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
        return {"stage3": outs[0], "stage4": outs[1]}, task_query, prompt_loss

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
        x = self.reid_box_pooler(features, roi_boxes)
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
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i + bonenum] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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
        fpn_features = self.sim_fpn(features)
        proposals, proposal_losses = self.proposal_generator(
            image_list, fpn_features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.swin,
            self.side_stage_prompts,
            task_query,
            image_list,
            fpn_features,
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

    def forward_gallery_vis(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        fpn_features = self.sim_fpn(features)
        proposals, proposal_losses = self.proposal_generator(
            image_list, fpn_features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            image_list,
            fpn_features,
            proposals,
            gt_instances,
        )
        roi_boxes = [inst.pred_boxes.tensor for inst in pred_instances]
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
            cate_idx = pred_scores.new_zeros(pred_scores.shape[0], dtype=torch.int64)
            # nms
            keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)
            pred_boxes_t = pred_boxes_t[keep]
            pred_scores = pred_scores[keep]
            pred_i.pred_boxes = Boxes(pred_boxes_t, BoxMode.XYXY_ABS)
            pred_i.pred_scores = pred_scores
            pred_i.reid_feats = pred_boxes_t  # trivial impl
            pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
            pred_i.assign_ids = pred_i.assign_ids[filter_mask][keep]
        return pred_instances, [feat.detach() for feat in features.values()]

    def forward_gallery_gt(self, image_list, gt_instances):
        assert not self.training
        # inf on faster rcnn boxes
        if self.pred_rst is None:
            pred_rst = torch.load(
                "Data/model_zoo/frcnn_prw_gallery_gt_inf.pt",
                map_location=self.device,
            )["infs"]
            self.pred_rst = {imgn: rst[:, :5] for imgn, rst in pred_rst.items()}
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        # roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        roi_boxes = [self.pred_rst[inst.image_id][:, :4] for inst in gt_instances]
        for i, gt_i in enumerate(gt_instances):
            # back to org scale
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            factor = torch.tensor(
                [[w / org_w, h / org_h, w / org_w, h / org_h]],
                dtype=torch.float32,
                device=self.device,
            )
            roi_boxes[i] = roi_boxes[i] * factor
        reid_feats = self.reid_head(task_query, image_list, features, roi_boxes)
        pred_instances = []
        for i, (gt_i, feats_i) in enumerate(zip(gt_instances, reid_feats)):
            inst = Instances(gt_i.image_size)
            inst.pred_boxes = Boxes(roi_boxes[i])  # gt_i.gt_boxes
            # inst.pred_scores = (
            #    torch.zeros_like(gt_i.gt_pids, dtype=feats_i.dtype) + 0.99
            # )
            inst.pred_scores = self.pred_rst[gt_i.image_id][:, 4]
            # inst.pred_classes =torch.zeros_like(gt_i.gt_pids)
            inst.pred_classes = torch.zeros_like(
                self.pred_rst[gt_i.image_id][:, 4], dtype=torch.int64
            )

            # inst.assign_ids =gt_i.gt_pids
            inst.assign_ids = torch.zeros_like(
                self.pred_rst[gt_i.image_id][:, 4], dtype=torch.int64
            )

            inst.reid_feats = feats_i
            # back to org scale
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            inst.pred_boxes.scale(org_w / w, org_h / h)
            pred_instances.append(inst)

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
class PromptedSwinF4RCNNPSBoxAug(PromptedSwinF4RCNNPS):
    @configurable
    def __init__(
        self,
        box_aug,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        self.box_aug = box_aug

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["box_aug"] = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        return ret

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
            reid_loss = self.reid_head(
                task_query, image_list, features, pos_boxes, pos_ids
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


@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNRCNNPSBoxAug(PromptedSwinSimFPNRCNNPS):
    @configurable
    def __init__(
        self,
        box_aug,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        self.box_aug = box_aug

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["box_aug"] = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        fpn_features = self.sim_fpn(features)
        proposals, proposal_losses = self.proposal_generator(
            image_list, fpn_features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            image_list,
            fpn_features,
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
            reid_loss = self.reid_head(
                task_query, image_list, features, pos_boxes, pos_ids
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


@META_ARCH_REGISTRY.register()
class PromptedMsSwinSimFPNLiteRCNNPSBoxAug(PromptedSwinSimFPNRCNNPSBoxAug):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        num_prompts = prompt_cfg.NUM_PROMPTS
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 4

        # NOTE downsample module of stage3 is trainable
        swin = ret["swin"]

        stage_prompts = nn.ModuleList()
        for si, nl in enumerate(tr_cfg.DEPTH):
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            if stage_num_prompts == 0:
                stage_prompts.append(nn.Identity())
                continue
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[si],
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
            stage_prompts.append(prompt_stage)
        side_stage_prompts = nn.ModuleList()
        if isinstance(num_prompts, int):
            stage_num_prompts = num_prompts
        else:
            stage_num_prompts = num_prompts[-1]
        if stage_num_prompts == 0:
            stage_prompts.append(nn.Identity())
        else:
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=swin.num_features[-1],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=swin.num_features[-1],
                    num_prompts=stage_num_prompts,
                    num_layers=tr_cfg.DEPTH[-1],
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=sum(swin.num_features[-4:]),
                    vis_period=cfg.VIS_PERIOD,
                )
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)
            side_stage_prompts.append(prompt_stage)

        ret.update(
            {
                "swin": swin,
                "stage_prompts": stage_prompts,
                "side_stage_prompts": side_stage_prompts,
            }
        )
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
        outs = []
        for i, stage in enumerate(self.swin_org.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin_org.semantic_weight >= 0:
                sw = self.swin_org.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin_org.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin_org.softplus(sw) + sb
            out = getattr(self.swin_org, "norm{}".format(i))(out)
            out = (
                out.view(-1, *out_hw_shape, self.swin_org.num_features[i])
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.swin_org.avgpool(out)
            out = torch.flatten(out, 1)
            outs.append(out)
        return torch.cat(outs, dim=-1)

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
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        bonenum = 3
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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
                outd = norm_layer(out)
                outd = (
                    outd.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(outd)  # for det
                norm_layer = getattr(self.swin, f"norm{i}")
                outr = norm_layer(out)
                outr = (
                    outr.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(outr)  # for det
        if self.training:
            prompt_loss = {"loss_prompt": prompt_loss}
        return {"side_stage3": outs[0], "stage3": outs[1]}, task_query, prompt_loss


@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNLiteRCNNPSBoxAug(PromptedMsSwinSimFPNLiteRCNNPSBoxAug):
    @classmethod
    def from_config(cls, cfg):
        return super(PromptedMsSwinSimFPNLiteRCNNPSBoxAug, cls).from_config(cfg)

    @torch.no_grad()
    def task_query(self, backbone_features):
        return super(PromptedMsSwinSimFPNLiteRCNNPSBoxAug, self).task_query(
            backbone_features
        )

@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNLiteRCNNPSBoxAug2c(PromptedSwinSimFPNLiteRCNNPSBoxAug):
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
        x = self.reid_box_pooler(features, roi_boxes)
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
        s_feats=[x.view(x.shape[0],*hw_shape,x.shape[-1]).permute(0,3,1,2).contiguous()]
        if self.swin.semantic_weight >= 0:
            sw = self.swin.semantic_embed_w[bonenum - 1](semantic_weight).unsqueeze(1)
            sb = self.swin.semantic_embed_b[bonenum - 1](semantic_weight).unsqueeze(1)
            x = x * self.swin.softplus(sw) + sb
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i + bonenum] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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
        s_feats.append(out)
        out=torch.cat(s_feats,dim=1)
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
@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNLiteRCNNPSBoxAug2s(PromptedSwinSimFPNLiteRCNNPSBoxAug):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.oim_loss_s3 =copy.deepcopy(self.oim_loss)
        self.bn_neck_s3=copy.deepcopy(self.bn_neck)
        self.pool_layer_s3=copy.deepcopy(self.pool_layer)
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
        x = self.reid_box_pooler(features, roi_boxes)
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
        x_s3=x.view(x.shape[0],*hw_shape,x.shape[-1]).permute(0,3,1,2).contiguous()
        embs_s3 = self.pool_layer_s3(x_s3).reshape((x_s3.shape[0], -1))
        embs_s3 = self.bn_neck_s3(embs_s3)
        if self.swin.semantic_weight >= 0:
            sw = self.swin.semantic_embed_w[bonenum - 1](semantic_weight).unsqueeze(1)
            sb = self.swin.semantic_embed_b[bonenum - 1](semantic_weight).unsqueeze(1)
            x = x * self.swin.softplus(sw) + sb
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        task_query_x = task_query.unsqueeze(1)
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i + bonenum] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
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
            reid_loss_s3=self.oim_loss_s3(embs_s3, pos_ids)
            reid_loss.update({k+"_s3":v for k,v in reid_loss_s3.items()})
            return reid_loss
        else:
            embs=torch.cat([embs_s3,embs],dim=-1)
            embs = normalize(embs, dim=-1)
            return torch.split(embs, [len(bxs) for bxs in roi_boxes])

@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNLiteRCNNPSBoxAugHeadOpen(PromptedSwinSimFPNLiteRCNNPSBoxAug):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        for p in self.roi_heads.parameters():
            p.requires_grad = True


@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNLiteRCNNPSBoxAugRpnOpen(PromptedSwinSimFPNLiteRCNNPSBoxAug):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        for p in self.roi_heads.parameters():
            p.requires_grad = True
        for p in self.proposal_generator.parameters():
            p.requires_grad = True


@META_ARCH_REGISTRY.register()
class PromptedSwinSimFPNLiteRCNNPSBoxAugFpnOpen(PromptedSwinSimFPNLiteRCNNPSBoxAug):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        for p in self.roi_heads.parameters():
            p.requires_grad = True
        for p in self.proposal_generator.parameters():
            p.requires_grad = True
        for p in self.sim_fpn.parameters():
            p.requires_grad = True


import torch.nn.functional as tF


@META_ARCH_REGISTRY.register()
class TiPromptedMsSwinSimFPNLiteRCNNPSBoxAug(PromptedSwinSimFPNRCNNPSBoxAug):
    @configurable
    def __init__(
        self,
        n_tasks,
        cur_task,
        task_keys,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        self.task_keys = nn.Parameter(task_keys, requires_grad=True)
        self.n_tasks = n_tasks
        self.cur_task = cur_task

        for n, p in self.roi_heads.named_parameters():
            if "ti" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.pred_rst = None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        inc_cfg = cfg.PERSON_SEARCH.INCREMENTAL
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 4

        # NOTE downsample module of stage3 is trainable
        swin = ret["swin"]

        stage_prompts = nn.ModuleList()
        for si, nl in enumerate(tr_cfg.DEPTH):
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            if stage_num_prompts == 0:
                stage_prompts.append(nn.Identity())
                continue
            assert prompt_cfg.PROMPT_TYPE == "FixedTaskInc"
            prompt_stage = prompts.FixedPromptsTaskInc(
                emb_d=swin.num_features[si],
                n_tasks=inc_cfg.NUM_TASKS,
                num_prompts=stage_num_prompts,
                num_layers=nl,
            )
            prompt_stage.process_task_count(inc_cfg.CUR_TASK)
            stage_prompts.append(prompt_stage)
        side_stage_prompts = nn.ModuleList()
        if isinstance(num_prompts, int):
            stage_num_prompts = num_prompts
        else:
            stage_num_prompts = num_prompts[-1]
        if stage_num_prompts == 0:
            stage_prompts.append(nn.Identity())
        else:
            prompt_stage = prompts.FixedPromptsTaskInc(
                emb_d=swin.num_features[-1],
                n_tasks=inc_cfg.NUM_TASKS,
                num_prompts=stage_num_prompts,
                num_layers=tr_cfg.DEPTH[-1],
            )
            prompt_stage.process_task_count(inc_cfg.CUR_TASK)
            side_stage_prompts.append(prompt_stage)
        task_keys = torch.FloatTensor(
            inc_cfg.NUM_TASKS, inc_cfg.LEN_KEYS, swin.num_features[-1]
        )
        nn.init.uniform_(task_keys)
        roi_heads = AlteredStandaredROIHeadsTi(cfg, ret["sim_fpn"].output_shape())
        ret.update(
            {
                "n_tasks": inc_cfg.NUM_TASKS,
                "cur_task": inc_cfg.CUR_TASK,
                "swin": swin,
                "stage_prompts": stage_prompts,
                "side_stage_prompts": side_stage_prompts,
                "task_keys": task_keys,
                "roi_heads": roi_heads,
            }
        )
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

    def swin_backbone(self, x, ti):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        x = x[list(x.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        bonenum = 3
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
                selected_prompts = self.stage_prompts[i](len(stage.blocks), ti)
                x, hw_shape, out, out_hw_shape = stage(
                    x, hw_shape, deep_prompt_embd=selected_prompts
                )
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum - 1:
                norm_layer = getattr(self.swin, f"side_norm{i}")
                outd = norm_layer(out)
                outd = (
                    outd.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(outd)  # for det
                norm_layer = getattr(self.swin, f"norm{i}")
                outr = norm_layer(out)
                outr = (
                    outr.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(outr)  # for det
        return {"side_stage3": outs[0], "stage3": outs[1]}

    def task_select(self, query):
        if self.training:
            keys = self.task_keys[self.cur_task]
            keys_n = tF.normalize(keys, dim=-1)
            query_n = tF.normalize(query, dim=-1)
            sim = torch.einsum("kj,bj->kb", keys_n, query_n)
            loss = (1 - sim).sum() / query.shape[0]
            ti = [self.cur_task] * query.shape[0]
            loss = {"loss_task_cos": loss}
        else:
            keys_n = tF.normalize(self.task_keys, dim=-1).flatten(0, 1)
            query_n = tF.normalize(query, dim=-1)
            cos_sim = torch.einsum("bj,kj->bk", query_n, keys_n)
            domain_ids = torch.cat(
                [
                    torch.zeros(self.task_keys.shape[1], device=self.device) + i
                    for i in range(self.task_keys.shape[0])
                ]
            ).long()
            ti = []
            for cos_sim_i in cos_sim:
                topk_idx = torch.topk(cos_sim_i, k=self.task_keys.shape[1])[1]
                di, ndi = torch.unique(domain_ids[topk_idx], return_counts=True)
                ti.append(di[torch.argmax(ndi)].item())
            loss = None
        return ti, loss

    def forward_query(self, image_list, gt_instances):
        bk = self.backbone(image_list.tensor)
        x_query = self.task_query(bk)
        ti, _ = self.task_select(x_query)
        features = self.swin_backbone(bk, ti)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(ti, image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]

    def forward_gallery(self, image_list, gt_instances):
        bk = self.backbone(image_list.tensor)
        x_query = self.task_query(bk)
        ti, t_loss = self.task_select(x_query)
        features = self.swin_backbone(bk, ti)
        fpn_features = self.sim_fpn(features)
        proposals, proposal_losses = self.proposal_generator(
            image_list, fpn_features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            ti,
            image_list,
            fpn_features,
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
            reid_loss = self.reid_head(ti, image_list, features, pos_boxes, pos_ids)
            losses.update(reid_loss)
            losses.update(t_loss)

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
            reid_feats = self.reid_head(ti, image_list, features, roi_boxes)
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

    def forward_gallery_gt(self, image_list, gt_instances):
        assert not self.training
        # inf on faster rcnn boxes
        if self.pred_rst is None:
            pred_rst = torch.load(
                "Data/model_zoo/frcnn_prw_gallery_gt_inf.pt",
                map_location=self.device,
            )["infs"]
            self.pred_rst = {imgn: rst[:, :5] for imgn, rst in pred_rst.items()}

        bk = self.backbone(image_list.tensor)
        x_query = self.task_query(bk)
        ti, _ = self.task_select(x_query)
        features = self.swin_backbone(bk, ti)
        # roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        roi_boxes = [self.pred_rst[inst.image_id][:, :4] for inst in gt_instances]
        for i, gt_i in enumerate(gt_instances):
            # back to org scale
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            factor = torch.tensor(
                [[w / org_w, h / org_h, w / org_w, h / org_h]],
                dtype=torch.float32,
                device=self.device,
            )
            roi_boxes[i] = roi_boxes[i] * factor
        reid_feats = self.reid_head(ti, image_list, features, roi_boxes)
        pred_instances = []
        for i, (gt_i, feats_i) in enumerate(zip(gt_instances, reid_feats)):
            inst = Instances(gt_i.image_size)
            inst.pred_boxes = Boxes(roi_boxes[i])  # gt_i.gt_boxes
            # inst.pred_scores = (
            #    torch.zeros_like(gt_i.gt_pids, dtype=feats_i.dtype) + 0.99
            # )
            inst.pred_scores = self.pred_rst[gt_i.image_id][:, 4]
            inst.pred_classes = torch.zeros_like(
                self.pred_rst[gt_i.image_id][:, 4], dtype=torch.int64
            )  # torch.zeros_like(gt_i.gt_pids)
            inst.assign_ids = torch.zeros_like(
                self.pred_rst[gt_i.image_id][:, 4], dtype=torch.int64
            )  # gt_i.gt_pids
            inst.reid_feats = feats_i
            # back to org scale
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            inst.pred_boxes.scale(org_w / w, org_h / h)
            pred_instances.append(inst)

        return pred_instances

    def reid_head(
        self, task_ids, image_list, backbone_features, roi_boxes, roi_ids=None
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
        x = self.reid_box_pooler(features, roi_boxes)
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
            if (
                not isinstance(self.swin.num_prompts, int)
                and self.swin.num_prompts[i + bonenum] == 0
            ):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
                selected_prompts = self.stage_prompts[bonenum + i](
                    len(stage.blocks), task_ids
                )
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
            return reid_loss
        else:
            embs = normalize(embs, dim=-1)
            return torch.split(embs, [len(bxs) for bxs in roi_boxes])


@META_ARCH_REGISTRY.register()
class PromptedSwinF4RCNNPSHybrid(PromptedSwinF4RCNNPS):
    @classmethod
    def from_config(cls, cfg):
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        ret = super().from_config(cfg)
        swin_org = SwinTransformer(
            semantic_weight=tr_cfg.SEMANTIC_WEIGHT_ORG,
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
        return ret


# TODO refactor new L2P code
@META_ARCH_REGISTRY.register()
class PrefixPromptedSwinF4RCNNPS(PromptedSwinF4RCNNPS):
    @configurable
    def __init__(
        self,
        box_aug,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        self.box_aug = box_aug

    @classmethod
    def from_config(cls, cfg):
        ret = super(SwinF4RCNNPS, cls).from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        if "L2P" in prompt_cfg.PROMPT_TYPE and "Attn" not in prompt_cfg.PROMPT_TYPE:
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
            elif prompt_cfg.PROMPT_TYPE == "L2POCos":
                prompt_stage = prompts.L2POrgCos(
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
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=swin.num_features[si],
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=prompt_cfg.NUM_PROMPTS,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=swin.num_features[-1],
                    vis_period=cfg.VIS_PERIOD,
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
            prompt_stage.process_task_count(prompt_cfg.CURRECT_TASK)

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
        elif prompt_cfg.PROMPT_TYPE == "L2Ppp":
            prompt_stage = prompts.L2Ppp(
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
        ret["oim_loss"] = (
            OIMLoss(cfg)
            if hasattr(cfg.PERSON_SEARCH.REID.LOSS, "OIM")
            else IncOIMLoss(cfg)
        )
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        roi_heads = AlteredStandaredROIHeadsTi(cfg, ret["sim_fpn"].output_shape())
        ret["roi_heads"] = roi_heads
        return ret


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
