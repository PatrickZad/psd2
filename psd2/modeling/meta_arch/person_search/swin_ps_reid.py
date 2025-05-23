# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as tF
from psd2.config import configurable

from psd2.structures import  Instances, Boxes, BoxMode
from psd2.utils.events import get_event_storage

from .base import SearchBase


from .. import META_ARCH_REGISTRY
from psd2.modeling.extend.solider import (
    SwinTransformer,
    PromptedSwinTransformer,
    PrefixPromptedSwinTransformer,
    SwinTransformer,
)
from psd2.modeling.prompts import build_stage_prompt_pool
from psd2.layers.metric_loss import TripletLoss
from torch.nn import init
from psd2.layers.mem_matching_losses import OIMLoss

from psd2.modeling.box_augmentation import build_box_augmentor
from psd2.modeling.poolers import ROIPooler
from psd2.layers import ShapeSpec
from .swin_rcnn_pd import OrgSwinF4AttnFPN


@META_ARCH_REGISTRY.register()
class SwinF4PSReid(SearchBase):
    """
    stage3 w/ norm as output, semantic=0.6 all the time
    """

    @configurable
    def __init__(
        self,
        swin,
        box_aug,
        roi_pooler,
        reid_in_feature,
        bn_neck,
        pool_layer,
        oim_loss,
        triplet_loss,
        *args,
        **kws,
    ):
        super().__init__(
            *args,
            **kws,
        )
        self.swin = swin
        self.box_aug = box_aug
        self.roi_pooler=roi_pooler
        self.reid_in_feature=reid_in_feature
        self.oim_loss=oim_loss
        self.triplet_loss=triplet_loss
        self.bn_neck=bn_neck
        self.pool_layer=pool_layer

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        # NOTE downsample module of stage3 is trainable
        swin = SwinTransformer(
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

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = tuple(
            1.0 / swin_out_shape[k].stride
            for k in [cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT]
        )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        assert not cfg.MODEL.KEYPOINT_ON

        ret["roi_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["box_aug"] = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        ret["swin"]= swin
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
        ret["triplet_loss"]= TripletLoss(0.3, "mean")
        ret["pool_layer"] = nn.AdaptiveAvgPool2d((1, 1))
        ret["reid_in_feature"] = cfg.PERSON_SEARCH.REID.MODEL.IN_FEAT
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features = self.swin_backbone(image_list.tensor)
        if self.training:
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
            losses = self.reid_head(image_list, features, pos_boxes, pos_ids)
            pred_instances=[]
            for i,(boxes_i,ids_i) in enumerate(zip(pos_boxes,pos_ids)):
                inst=Instances(gt_instances[i].image_size)
                inst.pred_boxes=Boxes(boxes_i, BoxMode.XYXY_ABS)
                inst.pred_scores=torch.zeros_like(ids_i,dtype=boxes_i.dtype)+0.99
                inst.pred_classes = torch.zeros_like(ids_i)
                inst.assign_ids=ids_i
                inst.reid_feats=boxes_i
                pred_instances.append(inst)

            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
            reid_feats=self.reid_head(image_list, features, roi_boxes)
            pred_instances=[]
            for i,(gt_i,feats_i) in enumerate(zip(gt_instances,reid_feats)):
                inst=Instances(gt_i.image_size)
                inst.pred_boxes=gt_i.gt_boxes
                inst.pred_scores=torch.zeros_like(gt_i.gt_pids,dtype=feats_i.dtype)+0.99
                inst.pred_classes = torch.zeros_like(gt_i.gt_pids)
                inst.assign_ids=gt_i.gt_pids
                inst.reid_feats=feats_i
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                inst.pred_boxes.scale(org_w / w, org_h / h)
                pred_instances.append(inst)
                
            return pred_instances
    def forward_query(self, image_list, gt_instances):
        features= self.swin_backbone(image_list.tensor)
        roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
        box_embs = self.reid_head(image_list, features, roi_boxes)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]
    def swin_backbone(self, x):
        # setting swin.semantic_weight == 0 makes it compatible with ign swins
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
                norm_layer = getattr(self.swin, f"norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return {"stage3": outs[-1]}

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
        x = self.roi_pooler(features, roi_boxes)
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
            losses = self.oim_loss(embs, pos_ids)
            """
            NOTE not consider dist training in triplet for now
            oim_lookup = self.oim_loss.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = embs[pos_ids > -1]
            feats2 = torch.cat([feats1, oim_lookup], dim=0)
            trip = self.triplet_loss(
                feats1,
                feats2,
                pos_ids[pos_ids > -1],
                torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                normalize_feature=True,
            )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k] = v
                else:
                    get_event_storage().put_scalar(k, v)
            """
            return losses
        else:
            embs = tF.normalize(embs, dim=-1)
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

@META_ARCH_REGISTRY.register()
class SwinF4PSReidLastStride(SwinF4PSReid):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin = SwinTransformer(
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
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / swin_out_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["roi_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["swin"]= swin
        return ret


@META_ARCH_REGISTRY.register()
class SwinF4PSReidFrozen(SwinF4PSReid):
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
        for p in self.parameters():
            p.requires_grad = False
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        for module in self.children():
            module.train(False)
@META_ARCH_REGISTRY.register()
class SwinF4PSReidWoLn(SwinF4PSReid):
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
        for i in range(len(self.swin.stages)):
            if hasattr(self.swin,f'norm{i}'):
                setattr(self.swin,f'norm{i}',nn.Identity())
@META_ARCH_REGISTRY.register()
class SwinF4PSReidFrozenWoLn(SwinF4PSReidFrozen):
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
        for i in range(len(self.swin.stages)):
            if hasattr(self.swin,f'norm{i}'):
                setattr(self.swin,f'norm{i}',nn.Identity())

@META_ARCH_REGISTRY.register()
class PromptedSwinF4PSReid(SwinF4PSReid):
    @configurable
    def __init__(
        self,
        stage_prompts,
        swin_org,
        swin_org_init_path,
        precomputed_box=None,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.stage_prompts = stage_prompts
        self.swin_org = swin_org
        self.swin_org_init_path = swin_org_init_path
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.swin.parameters():
            p.requires_grad = False
        for p in self.swin_org.parameters():
            p.requires_grad = False
        self.precomputed_box=precomputed_box

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
        res=self.swin_org.load_state_dict(state_dict, strict=False)
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
            self.stage_prompts.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = SwinF4PSReid.from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        if "L2P" in prompt_cfg.PROMPT_TYPE:
            if prompt_cfg.STAGE_WISE:
                num_prompts = [n  * prompt_cfg.TOP_K for n in prompt_cfg.NUM_PROMPTS]
            else:
                num_prompts = prompt_cfg.TOP_K * prompt_cfg.NUM_PROMPTS
        else:
            num_prompts = prompt_cfg.NUM_PROMPTS
        swin = PromptedSwinTransformer(
            prompt_start_stage=prompt_cfg.PROMPT_START_STAGE,
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
        stage_prompts = nn.ModuleList()
        for si, nl in enumerate(tr_cfg.DEPTH):
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            prompt_stage=build_stage_prompt_pool(prompt_cfg,stage_num_prompts,swin.num_features[si],nl,swin.num_features[-1],cfg.VIS_PERIOD)
            stage_prompts.append(prompt_stage)
        ret.update(
            {
                "swin": swin,
                "stage_prompts": stage_prompts,
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
        if hasattr(cfg.PERSON_SEARCH,"PRE_INFERENCE_RESULT"):
            inf_result=torch.load(cfg.PERSON_SEARCH.PRE_INFERENCE_RESULT,map_location="cpu")
            infs=inf_result["infs"]
            precomputed_box={}
            for k,v in infs.items():
                precomputed_box[k]=v[:,:5]
            ret["precomputed_box"]=precomputed_box
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
        x = self.roi_pooler(features, roi_boxes)
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
            losses = self.oim_loss(embs, pos_ids)
            """
            NOTE not consider dist training in triplet for now
            oim_lookup = self.oim_loss.lb_layer.lookup_table
            lookup_ids = torch.arange(
                0, oim_lookup.shape[0], dtype=pos_ids.dtype, device=pos_ids.device
            )
            feats1 = embs[pos_ids > -1]
            feats2 = torch.cat([feats1, oim_lookup], dim=0)
            trip = self.triplet_loss(
                feats1,
                feats2,
                pos_ids[pos_ids > -1],
                torch.cat([pos_ids[pos_ids > -1], lookup_ids], dim=0),
                normalize_feature=True,
            )
            for k, v in trip.items():
                if "loss" in k:
                    losses[k] = v
                else:
                    get_event_storage().put_scalar(k, v)
            """
            return losses
        else:
            embs = tF.normalize(embs, dim=-1)
            return torch.split(embs, [len(bxs) for bxs in roi_boxes])

    def forward_gallery(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        if self.training:
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
            losses = self.reid_head(task_query,image_list, features, pos_boxes, pos_ids)
            losses.update(prompt_loss)
            pred_instances=[]
            for i,(boxes_i,ids_i) in enumerate(zip(pos_boxes,pos_ids)):
                inst=Instances(gt_instances[i].image_size)
                inst.pred_boxes=Boxes(boxes_i, BoxMode.XYXY_ABS)
                inst.pred_scores=torch.zeros_like(ids_i,dtype=boxes_i.dtype)+0.99
                inst.pred_classes = torch.zeros_like(ids_i)
                inst.assign_ids=ids_i
                inst.reid_feats=boxes_i
                pred_instances.append(inst)

            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            if self.precomputed_box is not None:
                roi_boxes=[]
                scores=[]
                for gt_i in gt_instances:
                    inf_boxes=Boxes(self.precomputed_box[gt_i.image_id][:,:4])
                    scores.append(self.precomputed_box[gt_i.image_id][:,4])
                    org_h, org_w = gt_i.org_img_size
                    h, w = gt_i.image_size
                    inf_boxes.scale(w/org_w,h/org_h)
                    gt_boxes_i=gt_i.gt_boxes
                    roi_boxes.append(inf_boxes.tensor.to(self.device))
            else:
                roi_boxes = [inst.gt_boxes.tensor for inst in gt_instances]
            reid_feats=self.reid_head(task_query,image_list, features, roi_boxes)
            pred_instances=[]
            if self.precomputed_box is not None:
                for i,(gt_i,feats_i) in enumerate(zip(gt_instances,reid_feats)):
                    inst=Instances(gt_i.image_size)
                    inst.pred_boxes=Boxes(roi_boxes[i])
                    inst.pred_scores=scores[i].to(roi_boxes[i].device)
                    inst.pred_classes = torch.zeros(roi_boxes[i].shape[0],dtype=gt_i.gt_pids.dtype)
                    inst.assign_ids=torch.zeros(roi_boxes[i].shape[0],dtype=gt_i.gt_pids.dtype)
                    inst.reid_feats=feats_i
                    # back to org scale
                    org_h, org_w = gt_i.org_img_size
                    h, w = gt_i.image_size
                    inst.pred_boxes.scale(org_w / w, org_h / h)
                    pred_instances.append(inst)
            else:
                for i,(gt_i,feats_i) in enumerate(zip(gt_instances,reid_feats)):
                    inst=Instances(gt_i.image_size)
                    inst.pred_boxes=gt_i.gt_boxes
                    inst.pred_scores=torch.zeros_like(gt_i.gt_pids,dtype=feats_i.dtype)+0.99
                    inst.pred_classes = torch.zeros_like(gt_i.gt_pids)
                    inst.assign_ids=gt_i.gt_pids
                    inst.reid_feats=feats_i
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
class PromptedSwinF4PSReidLastStride(PromptedSwinF4PSReid):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        if "L2P" in prompt_cfg.PROMPT_TYPE:
            num_prompts = prompt_cfg.NUM_PROMPTS * prompt_cfg.TOP_K
        else:
            num_prompts = prompt_cfg.NUM_PROMPTS
        swin = PromptedSwinTransformer(
            prompt_start_stage=prompt_cfg.PROMPT_START_STAGE,
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
            strides=(4, 2, 2, 1),
        )
        
        swin_out_shape = {
            "stage{}".format(i + 1): ShapeSpec(
                channels=swin.num_features[i], stride=swin.strides[i]
            )
            for i in range(len(swin.stages))
        }
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / swin_out_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["roi_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["swin"]= swin
        
        return ret

@META_ARCH_REGISTRY.register()
class PrefixPromptedSwinF4PSReid(PromptedSwinF4PSReid):
    @classmethod
    def from_config(cls, cfg):
        ret = PromptedSwinF4PSReid.from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        if "L2P" in prompt_cfg.PROMPT_TYPE:
            num_prompts = prompt_cfg.NUM_PROMPTS * prompt_cfg.TOP_K
        else:
            num_prompts = prompt_cfg.NUM_PROMPTS
        swin = PrefixPromptedSwinTransformer(
            prompt_start_stage=prompt_cfg.PROMPT_START_STAGE,
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
        ret["swin"]=swin
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

from ...proposal_generator import build_proposal_generator
from psd2.modeling.id_assign import build_id_assigner
from .swin_rcnn_pd import LastLevelMaxPool,AlteredStandaredROIHeads,PromptedAttnFeaturePyramid
from psd2.layers import batched_nms
from torch.utils.checkpoint import checkpoint as ckpt

@META_ARCH_REGISTRY.register()
class PromptedSwinF4PSReidSGAttnFPN(PromptedSwinF4PSReid):
    @configurable
    def __init__(
        self,
        attn_fpn,
        fpn_prompts,
        proposal_generator,
        roi_heads,
        id_assigner,
        *args,
        **kws,
    ):
        PromptedSwinF4PSReid.__init__(
            self,*args,**kws
        )
        self.attn_fpn = attn_fpn
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.id_assigner = id_assigner
        self.fpn_prompts=fpn_prompts
        for p in self.attn_fpn.parameters():  # norm2 is not supervised in solider
            p.requires_grad = False
        for p in self.proposal_generator.parameters():
            p.requires_grad = False
        for p in self.roi_heads.parameters():
            p.requires_grad = False
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.eval()
            self.swin.eval()
            self.swin_org.eval()
            self.stage_prompts.train()
            self.fpn_prompts.train()
            self.attn_fpn.train()
            self.proposal_generator.train()
            self.roi_heads.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    def load_state_dict(self, *args, **kws):
        return PromptedSwinF4PSReid.load_state_dict(self,*args, **kws)
    @classmethod
    def from_config(cls, cfg):
        ret=PromptedSwinF4PSReid.from_config(cfg)
        swin=ret["swin"]
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
        attn_fpn_cfg = cfg.PERSON_SEARCH.DET.MODEL.ATTN_FPN
        p5_m=LastLevelMaxPool()
        p5_m.in_feature="p4"
        if hasattr(attn_fpn_cfg,"WITH_CP"):
            with_cp=attn_fpn_cfg.WITH_CP
        else:
            with_cp=False
        attn_fpn = PromptedAttnFeaturePyramid(
            swin_out_shape,
            attn_fpn_cfg.IN_FEATURES,
            attn_fpn_cfg.OUT_CHANNELS,
            top_block=p5_m,
            with_cp=with_cp
        )
        roi_heads = AlteredStandaredROIHeads(cfg, attn_fpn.output_shape())
        ret["id_assigner"] = build_id_assigner(cfg)
        ret.update(
            {
                "proposal_generator": build_proposal_generator(
                    cfg, attn_fpn.output_shape()
                ),
                "roi_heads": roi_heads,
                "swin": swin,
                "attn_fpn": attn_fpn,
            }
        )
        ret["box_aug"] = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 3
        
        fpn_prompts = nn.ModuleList()
        for si in range(len(tr_cfg.DEPTH)-1):
            if isinstance(num_prompts, int):
                stage_num_prompts = num_prompts
            else:
                stage_num_prompts = num_prompts[si]
            prompt_stage=build_stage_prompt_pool(prompt_cfg,stage_num_prompts,swin.num_features[si],1,swin.num_features[-1],cfg.VIS_PERIOD)
            fpn_prompts.append(prompt_stage)
        ret["fpn_prompts"]=fpn_prompts
        return ret
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

        outs = {}
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        prompt_loss_d = torch.zeros(
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
            norm_layer = getattr(self.swin, f"norm{i}")
            outd = norm_layer(out).detach()
            outd = (
                    outd.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            task_query_d = task_query_x.expand(-1, 1, -1)
            selected_prompts_d, p_loss_d = self.fpn_prompts[i](
                    task_query_d, "p{}".format(i+2), train=self.training
                )
            outs["side_stage{}".format(i+1)]=outd
            outs["side_stage{}_prompts".format(i+1)]=selected_prompts_d[0] # 1 layer
            prompt_loss_d += p_loss_d
            if i==bonenum-1:
                outr = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs["stage3"]=outr
        if self.training:
            prompt_loss = {"loss_prompt": prompt_loss,"loss_prompts_det":prompt_loss_d}
        return outs, task_query, prompt_loss
    def forward_gallery(self, image_list, gt_instances):
        features, task_query, prompt_loss = self.swin_backbone(image_list.tensor)
        fpn_features = self.attn_fpn(features)
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
            reid_loss = self.reid_head(task_query,image_list, features, pos_boxes, pos_ids)
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
            return pred_instances, [feat.detach() for feat in features.values() if feat.dim()==4], losses
        else:
            if self.precomputed_box is not None:
                roi_boxes=[]
                scores=[]
                for gt_i in gt_instances:
                    inf_boxes=Boxes(self.precomputed_box[gt_i.image_id][:,:4])
                    scores.append(self.precomputed_box[gt_i.image_id][:,4])
                    org_h, org_w = gt_i.org_img_size
                    h, w = gt_i.image_size
                    inf_boxes.scale(w/org_w,h/org_h)
                    gt_boxes_i=gt_i.gt_boxes
                    roi_boxes.append(inf_boxes.tensor.to(self.device))
            
            reid_feats=self.reid_head(task_query,image_list, features, roi_boxes)
            
            if self.precomputed_box is not None:
                pred_instances=[]
                for i,(gt_i,feats_i) in enumerate(zip(gt_instances,reid_feats)):
                    inst=Instances(gt_i.image_size)
                    inst.pred_boxes=Boxes(roi_boxes[i])
                    inst.pred_scores=scores[i].to(roi_boxes[i].device)
                    inst.pred_classes = torch.zeros(roi_boxes[i].shape[0],dtype=gt_i.gt_pids.dtype)
                    inst.assign_ids=torch.zeros(roi_boxes[i].shape[0],dtype=gt_i.gt_pids.dtype)
                    inst.reid_feats=feats_i
                    # back to org scale
                    org_h, org_w = gt_i.org_img_size
                    h, w = gt_i.image_size
                    inst.pred_boxes.scale(org_w / w, org_h / h)
                    pred_instances.append(inst)
            else:
                for i,(gt_i,feats_i) in enumerate(zip(gt_instances,reid_feats)):
                    inst=pred_instances[i]
                    inst.reid_feats=feats_i
                    # back to org scale
                    org_h, org_w = gt_i.org_img_size
                    h, w = gt_i.image_size
                    inst.pred_boxes.scale(org_w / w, org_h / h)

            return pred_instances
@META_ARCH_REGISTRY.register()
class PromptedSwinF4PSReidAttnFPN(PromptedSwinF4PSReidSGAttnFPN):
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

        outs = {}
        prompt_loss = torch.zeros(
            (1,), dtype=task_query.dtype, device=task_query.device
        )
        prompt_loss_d = torch.zeros(
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
            norm_layer = getattr(self.swin, f"norm{i}")
            outd = norm_layer(out) # .detach()
            outd = (
                    outd.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            task_query_d = task_query_x.expand(-1, 1, -1)
            selected_prompts_d, p_loss_d = self.fpn_prompts[i](
                    task_query_d, "p{}".format(i+2), train=self.training
                )
            outs["side_stage{}".format(i+1)]=outd
            outs["side_stage{}_prompts".format(i+1)]=selected_prompts_d[0] # 1 layer
            prompt_loss_d += p_loss_d
            if i==bonenum-1:
                outr = (
                    out.view(-1, *out_hw_shape, self.swin.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs["stage3"]=outr
        if self.training:
            prompt_loss = {"loss_prompt": prompt_loss,"loss_prompts_det":prompt_loss_d}
        return outs, task_query, prompt_loss