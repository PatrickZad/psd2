# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np

import torch
from torch import nn
import torch.nn.functional as tF
from psd2.config import configurable

from psd2.structures import Instances, Boxes, BoxMode
from psd2.utils.events import get_event_storage
from psd2.layers import batched_nms


from ...proposal_generator import build_proposal_generator
from .. import META_ARCH_REGISTRY

from psd2.modeling.id_assign import build_id_assigner
import psd2.modeling.prompts as prompts

from torch.nn import init
from psd2.layers.mem_matching_losses import OIMLoss,IncOIMLoss
from torch.nn.functional import normalize
from psd2.modeling import ShapeSpec
from psd2.modeling.poolers import ROIPooler
from .mvit_rcnn_pd import MViTF4RCNN,MViTSimFPNRCNN,SimpleFeaturePyramid,LastLevelMaxPool,AlteredStandaredROIHeads
from psd2.modeling.extend.mvit import MViT,SidePromptedMViTLS,get_abs_pos

# NOTE to test finetune and upper-bound
@META_ARCH_REGISTRY.register()
class MViTF4RCNNPS(MViTF4RCNN):
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
        num_stage3_blocks=self.trans._last_block_indexes[-2]-self.trans._last_block_indexes[-3]
        for n, p in self.trans.named_parameters():  
            if n.startswith("side_blocks."):
                bi=int(n.split(".")[1])
                if bi < num_stage3_blocks:
                    p.requires_grad = True
            else:
                if (
                    not n.startswith("side_norm3")
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
        start_i=self.trans._last_block_indexes[-2]+1
        for i, blk in enumerate(self.trans.blocks[start_i:]):
            x = blk(x)
            if i == self.trans._last_block_indexes[-1]:
                norm_layer = getattr(self.swin, f"scale5_norm")
                out = norm_layer(out)
                out = out.permute(0, 3, 1, 2).contiguous()
                
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
        features = self.trans_backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, losses, pos_match_indices = self.roi_heads(
            self.trans,image_list, features, proposals, gt_instances
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
        features = self.trans_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]



@META_ARCH_REGISTRY.register()
class MViTSimFPNRCNNPS(MViTSimFPNRCNN):
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
        num_stage3_blocks=self.trans._last_block_indexes[-2]-self.trans._last_block_indexes[-3]
        for n, p in self.trans.named_parameters():  
            if n.startswith("side_blocks."):
                bi=int(n.split(".")[1])
                if bi < num_stage3_blocks:
                    p.requires_grad = True
            else:
                if (
                    not n.startswith("side_norm3")
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
        patch_embed = ret["backbone"]
        trans_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=patch_embed.embed_dim*2**i, stride=4*2**(i if i<3 else i-1)
            )
            for i in range(4)
        }
        trans_out_shape.update(
            {
                "side_stage{}".format(i + 1): ShapeSpec(
                channels=patch_embed.embed_dim*2**i, stride=4*2**(i if i<3 else i-1)
            )
                for i in range(4)
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
            1.0 / trans_out_shape[k].stride
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
        raise NotImplementedError
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
        start_i=self.trans._last_block_indexes[-2]+1
        for i, blk in enumerate(self.trans.blocks[start_i:]):
            x = blk(x)
            if i == self.trans._last_block_indexes[-1]:
                norm_layer = getattr(self.swin, f"scale5_norm")
                out = norm_layer(out)
                out = out.permute(0, 3, 1, 2).contiguous()
                
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
        features = self.trans_backbone(image_list.tensor)
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
        features = self.trans_backbone(image_list.tensor)
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
        features = self.trans_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]

    def trans_backbone(self, x):
        raise NotImplementedError
        x = self.backbone(x)
        x = x[list(x.keys())[-1]]
        if self.trans.pos_emmbed is not None:
            x = x + get_abs_pos(self.trans.pos_embed, self.trans.pretrain_use_cls_token, x.shape[1:3])
        outs = []
        for i, blk in enumerate(self.trans.blocks):
            if i<=self.trans._last_block_indexes[-2]:
                x = blk(x)
                if i == self.trans._last_block_indexes[-2]:
                    x_out = self.trans.side_norm2(x)
                    outs.append(x_out.permute(0, 3, 1, 2))
            else:
                x=self.trans.side_block[i-(self.trans._last_block_indexes[-2]+1)](x)
                if i == self.trans._last_block_indexes[-1]:
                    x_out = self.trans.side_norm3(x)
                    outs.append(x_out.permute(0, 3, 1, 2))

        return {"stage3": outs[0], "stage4": outs[1]}

from psd2.modeling.extend.utils import get_abs_pos

@META_ARCH_REGISTRY.register()
class PromptedMViTSimFPNRCNNPS(MViTSimFPNRCNNPS):
    @configurable
    def __init__(
        self,
        stage_prompts,
        side_stage_prompts,
        trans_org,
        trans_org_init_path,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.stage_prompts = stage_prompts
        self.side_stage_prompts = side_stage_prompts
        self.trans_org = trans_org
        self.trans_org_init_path = trans_org_init_path
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.trans.parameters():
            p.requires_grad = False
        for p in self.trans_org.parameters():
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
        state_dict = _load_file(self.trans_org_init_path)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        org_ks = list(state_dict.keys())
        for k in org_ks:
            v = state_dict.pop(k)
            if k.startswith("trans."):
                nk = k[len("trans.") :]
                if not isinstance(v, torch.Tensor):
                    state_dict[nk] = torch.tensor(v, device=self.device)
                else:
                     state_dict[nk] =v
        res = self.trans_org.load_state_dict(state_dict, strict=False)
        print("parameters of *mvit_org* haved been loaded: \n")
        print(res)
        return out

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.eval()
            self.trans.eval()
            self.trans_org.eval()
            self.proposal_generator.train()  # to train the early prompts via rpn loss
            for m in self.proposal_generator.modules():
                if isinstance(m,torch.nn.BatchNorm2d):
                    m.eval()
            self.roi_heads.train()  # to train the prompts via det loss
            for m in self.roi_heads.modules():
                if isinstance(m,torch.nn.BatchNorm2d):
                    m.eval()
            self.stage_prompts.train()
            self.side_stage_prompts.train()
            self.sim_fpn.train()
            for m in self.sim_fpn.modules():
                if isinstance(m,torch.nn.BatchNorm2d):
                    m.eval()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = super(MViTSimFPNRCNNPS, cls).from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 4
        # NOTE downsample module of stage3 is trainable
        trans = SidePromptedMViTLS(
            side_start_stage=3,
            img_size=patch_embed.pretrain_img_size[0],
            patch_stride=(patch_embed.stride,patch_embed.stride)
            if isinstance(patch_embed.stride, int)
            else patch_embed.stride,
            embed_dim=patch_embed.embed_dim,
            depth=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            last_block_indexes=tr_cfg.LAST_BLOCK_INDEXS,
            qkv_pool_kernel=tr_cfg.QKV_POOL_KERNEL,
            adaptive_kv_stride=tr_cfg.ADAPTIVE_KV_STRIDE,
            adaptive_window_size=tr_cfg.ADAPTIVE_WINDOW_SIZE,
            residual_pooling=tr_cfg.RESIDUAL_POOLING,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            drop_path_rate=tr_cfg.DROP_PATH,
            with_cp=tr_cfg.WITH_CP, 
        )
        trans_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=patch_embed.embed_dim*2**i, stride=4*2**(i if i<3 else i-1)
            )
            for i in range(4)
        }
        trans_out_shape.update(
            {
                "side_stage{}".format(i + 1): ShapeSpec(
                channels=patch_embed.embed_dim*2**i, stride=4*2**(i if i<3 else i-1)
            )
                for i in range(4)
            }
        )
        sim_fpn_cfg = cfg.PERSON_SEARCH.DET.MODEL.SIM_FPN
        sim_fpn = SimpleFeaturePyramid(
            trans_out_shape,
            sim_fpn_cfg.IN_FEATURE,
            sim_fpn_cfg.OUT_CHANNELS,
            sim_fpn_cfg.SCALE_FACTORS,
            top_block=LastLevelMaxPool(),
        )
        roi_heads = AlteredStandaredROIHeads(cfg, sim_fpn.output_shape())
        ret["id_assigner"] = build_id_assigner(cfg)
        stage_prompts = nn.ModuleList()
        for si in range(len(tr_cfg.LAST_BLOCK_INDEXS)):
            if si==0:
                nl=tr_cfg.LAST_BLOCK_INDEXS[0]+1
            else:
                nl=tr_cfg.LAST_BLOCK_INDEXS[si]-tr_cfg.LAST_BLOCK_INDEXS[si-1]
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            if stage_num_prompts == 0:
                stage_prompts.append(nn.Identity())
                continue
            #NOTE attention modules do not change the dim 
            input_dim=trans._out_feature_channels[f"scale{si+2}"] # if si==0 else trans._out_feature_channels[si]//2
            prompt_embed_dim=input_dim*2*tr_cfg.NHEAD # seperate kp, vp
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=prompt_embed_dim,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
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
            input_dim=trans._out_feature_channels["scale5"]
            prompt_embed_dim=input_dim*2*tr_cfg.NHEAD # seperate kp, vp
            nl=tr_cfg.LAST_BLOCK_INDEXS[-1]-tr_cfg.LAST_BLOCK_INDEXS[-2]
            if prompt_cfg.PROMPT_TYPE == "L2Ppp":
                prompt_stage = prompts.L2Ppp(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask":
                prompt_stage = prompts.L2PppMask(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskAttn":
                prompt_stage = prompts.L2PppMaskAttn(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskBs":
                prompt_stage = prompts.L2PppMaskBs(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMask2":
                prompt_stage = prompts.L2PppMask2(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskM":
                prompt_stage = prompts.L2PppMaskM(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "L2PppMaskMC":
                prompt_stage = prompts.L2PppMaskMC(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            elif prompt_cfg.PROMPT_TYPE == "Fixed":
                prompt_stage = prompts.FixedPrompts(
                    emb_d=prompt_embed_dim,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                )
            elif prompt_cfg.PROMPT_TYPE == "CODAPromptWd":
                prompt_stage = prompts.CodaPromptWd(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
                    vis_period=cfg.VIS_PERIOD,
                )
            else:  # CODAPrompt
                prompt_stage = prompts.CodaPrompt(
                    emb_d=prompt_embed_dim,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=nl,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=trans._out_feature_channels["scale5"],
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
                "trans": trans,
                "sim_fpn": sim_fpn,
                "stage_prompts": stage_prompts,
                "side_stage_prompts": side_stage_prompts,
            }
        )
        trans_org = MViT(
            img_size=patch_embed.pretrain_img_size[0],
            patch_stride=(patch_embed.stride,patch_embed.stride)
            if isinstance(patch_embed.stride, int)
            else patch_embed.stride,
            embed_dim=patch_embed.embed_dim,
            depth=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            last_block_indexes=tr_cfg.LAST_BLOCK_INDEXS,
            qkv_pool_kernel=tr_cfg.QKV_POOL_KERNEL,
            adaptive_kv_stride=tr_cfg.ADAPTIVE_KV_STRIDE,
            adaptive_window_size=tr_cfg.ADAPTIVE_WINDOW_SIZE,
            residual_pooling=tr_cfg.RESIDUAL_POOLING,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            drop_path_rate=tr_cfg.DROP_PATH,
            with_cp=tr_cfg.WITH_CP,
        )
        ret["trans_org"] = trans_org
        ret["trans_org_init_path"] = cfg.PERSON_SEARCH.QUERY_ENCODER_WEIGHTS
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
            1.0 / trans_out_shape[k].stride
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
    def task_query(self, x):
        x=self.trans_org(x)["scale5"]
        x = tF.adaptive_avg_pool2d(x,(1,1))
        x = x.reshape(x.shape[0],-1)
        return x

    def trans_backbone(self, x):
        raise NotImplementedError
        x = self.backbone(x)
        task_query = self.task_query(x)
        task_query_x = task_query.unsqueeze(1)
        if self.trans.pos_embed is not None:
            x = x + get_abs_pos(self.trans.pos_embed, self.trans.pretrain_use_cls_token, x.shape[1:3])
        # all stage prompts
        si_prompts=[]
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        for si in range(3):
            num_stage_blocks=self.trans._last_block_indexes[si+1]-self.trans._last_block_indexes[si]
            task_query_stage = task_query_x.expand(-1, num_stage_blocks, -1)
            stage_prompts,pi_loss=self.stage_prompts[si](
                    task_query_stage, si, train=self.training
                ) # layers * b * L * c
            prompt_loss+=pi_loss
            si_prompts.append(stage_prompts)
        #side prompts
        num_stage_blocks=len(self.trans.side_blocks)
        task_query_stage = task_query_x.expand(-1, num_stage_blocks, -1)
        side_stage_prompts,pi_loss=self.stage_prompts[si](
                    task_query_stage, si, train=self.training
                ) # layers * b * L * c
        prompt_loss+=pi_loss
        
        len_bk=self.trans._last_block_indexes[-2]+1
        si=0
        outs=[]
        for i, blk in enumerate(self.trans.blocks):
            if i<=self.trans._last_block_indexes[-2]:
                li=0 if si==0 else i-(self.trans._last_block_indexes[si-1]+1)
                stage_prompts=si_prompts[si]
                x = blk(x,stage_prompts[li])
                if i==self.trans._last_block_indexes[-2]:
                    outs.append(x.permute(0, 3, 1, 2).contiguous())
            else:
                x=self.trans.side_blocks[i-len_bk](x,side_stage_prompts[i-len_bk])
                outs.append(self.trans.side_norm3(x).permute(0, 3, 1, 2).contiguous())
            for lsi,bi in enumerate(self.trans._last_block_indexes):
                if i==bi:
                    si=lsi
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
        x=x.permute(0,2,3,1).contiguous()
        del features

        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        task_query_x = task_query.unsqueeze(1)
        num_stage_blocks=self.trans._last_block_indexes[-1]-self.trans._last_block_indexes[-2]
        task_query_stage = task_query_x.expand(-1, num_stage_blocks, -1)
        if not isinstance(self.stage_prompts[-1],nn.Identity):
            last_stage_prompts,prompt_loss=self.stage_prompts[-1](
                        task_query_stage, f"reid_0", train=self.training
                    ) # layers * b * L * c
        last_stage_start=self.trans._last_block_indexes[-2]+1
        for i, blk in enumerate(self.trans.blocks[last_stage_start:]):
            if isinstance(self.stage_prompts[-1],nn.Identity):
                x = blk(x,None)
            else:
                expanded_prompts = []
                for bi in range(len(roi_boxes)):
                    expanded_prompts.append(
                        last_stage_prompts[i, bi : bi + 1].expand(
                            len(roi_boxes[bi]), -1, -1
                        )
                    )
                selected_prompts = torch.cat(expanded_prompts, dim=0)
                del expanded_prompts
                x=blk(x,selected_prompts)
            if i == num_stage_blocks - 1:
                norm_layer = getattr(self.trans, f"scale5_norm")
                out = norm_layer(x)
                out = out.permute(0, 3, 1, 2).contiguous() # bchw
                
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
        raise NotImplementedError

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
        features, task_query, _ = self.trans_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(task_query, image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]


@META_ARCH_REGISTRY.register()
class PromptedMViTSimFPNLiteRCNNPS(PromptedMViTSimFPNRCNNPS):
    def trans_backbone(self, x):
        x = self.backbone(x)
        x = x[list(x.keys())[-1]].permute(0,2,3,1).contiguous()
        task_query = self.task_query(x)
        task_query_x = task_query.unsqueeze(1)
        if self.trans.pos_embed is not None:
            x = x + get_abs_pos(self.trans.pos_embed, self.trans.pretrain_use_cls_token, x.shape[1:3])
        # all stage prompts
        si_prompts=[]
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        for si in range(3):
            num_stage_blocks=(self.trans._last_block_indexes[0]+1) if si==0 else self.trans._last_block_indexes[si]-self.trans._last_block_indexes[si-1]
            task_query_stage = task_query_x.expand(-1, num_stage_blocks, -1)
            stage_prompts,pi_loss=self.stage_prompts[si](
                    task_query_stage, si, train=self.training
                ) # layers * b * L * c
            prompt_loss+=pi_loss
            si_prompts.append(stage_prompts)
        
        si=0
        outs=[]
        for i, blk in enumerate(self.trans.blocks):
            li=i if si==0 else i-(self.trans._last_block_indexes[si-1]+1)
            stage_prompts=si_prompts[si]
            x = blk(x,stage_prompts[li])
            if i==self.trans._last_block_indexes[-2]:
                outs.append(x.permute(0, 3, 1, 2).contiguous())
                outs.append(self.trans.side_norm2(x).permute(0, 3, 1, 2).contiguous())
                break

            for lsi,bi in enumerate(self.trans._last_block_indexes):
                if i==bi:
                    si=lsi+1
        if self.training:
            prompt_loss = {"loss_prompt": prompt_loss}
        return {"stage3": outs[0], "side_stage3": outs[1]}, task_query, prompt_loss
    def forward_gallery(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.trans_backbone(image_list.tensor)
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


from psd2.modeling.box_augmentation import build_box_augmentor
import copy



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
