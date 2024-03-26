import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from .base_tbps import SearchBaseTBPS
from ..build import META_ARCH_REGISTRY
from psd2.utils.events import get_event_storage
from psd2.config.config import configurable
from psd2.modeling.roi_heads.roi_heads import (
    Res5ROIHeads,
    add_ground_truth_to_proposals,
)
from psd2.structures import Boxes, Instances, ImageList, pairwise_iou, BoxMode
from psd2.layers import ShapeSpec, batched_nms,FrozenBatchNorm2d
from psd2.utils.events import get_event_storage
from psd2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from psd2.modeling.proposal_generator import build_proposal_generator
from psd2.modeling.id_assign import build_id_assigner
from psd2.layers.mem_matching_losses import OIMLoss
from psd2.modeling.extend.clip_model import build_CLIP_from_openai_pretrained,Transformer,ResidualAttentionBlock
import copy
from collections import OrderedDict
import torch.utils.checkpoint as ckpt
from psd2.modeling.backbone import FPN

logger = logging.getLogger(__name__)

# GeneralizedRCNN as reference
@META_ARCH_REGISTRY.register()
class OimClip(SearchBaseTBPS):
    @configurable
    def __init__(
        self,
        clip_model,
        freeze_at_stage2,
        proposal_generator,
        roi_heads,
        id_assigner,
        bn_neck,
        oim_loss,
        oim_loss_text,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.clip_model=clip_model
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.id_assigner = id_assigner
        self.bn_neck = bn_neck
        self.bn_neck_text=copy.deepcopy(bn_neck)
        self.oim_loss = oim_loss
        self.oim_loss_text=oim_loss_text
        self.alignment_loss=SupConLoss()

        # res5 only in roi_head
        for p in self.clip_model.visual.layer4.parameters():
            p.requires_grad=False
        visual_encoder=self.clip_model.visual
        if freeze_at_stage2:
            for m in [visual_encoder.conv1, visual_encoder.conv2,  visual_encoder.conv3, visual_encoder.layer1]:
                for p in m.parameters():
                    p.requires_grad=False
            visual_encoder.bn1=convert_frozen_bn(visual_encoder.bn1)
            visual_encoder.bn2=convert_frozen_bn(visual_encoder.bn2)
            visual_encoder.bn3=convert_frozen_bn(visual_encoder.bn3)
            visual_encoder.layer1=convert_frozen_bn(visual_encoder.layer1)
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":16}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4)
        stride_block=res5[0]
        stride_block.avgpool=nn.Identity()
        stride_block.downsample[0]=nn.Identity()
        ret["roi_heads"] = ClipRes5ROIHeadsPs(cfg, res5,res_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
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
        oim_text=copy.deepcopy(ret["oim_loss"])
        del oim_text.ulb_layer
        oim_text.ulb_layer=None
        ret["oim_loss_text"]=oim_text
        return ret
    def backbone(self,x):
        visual_encoder=self.clip_model.visual
        def stem(x):
            for conv, bn in [(visual_encoder.conv1, visual_encoder.bn1), (visual_encoder.conv2, visual_encoder.bn2), (visual_encoder.conv3, visual_encoder.bn3)]:
                x = visual_encoder.relu(bn(conv(x)))
            x = visual_encoder.avgpool(x)
            return x
        # x = ckpt.checkpoint(stem,x)
        x=stem(x)
        x =ckpt.checkpoint(visual_encoder.layer1,x) if self.training else visual_encoder.layer1(x)
        # x =visual_encoder.layer1(x)
        x =ckpt.checkpoint(visual_encoder.layer2,x) if self.training else visual_encoder.layer2(x)
        x =ckpt.checkpoint(visual_encoder.layer3,x) if self.training else visual_encoder.layer3(x)
        return {"res4":x}
    def img_embes(self,roi_feats):
        return self.clip_model.visual.attnpool(roi_feats)
    def text_embeds(self,text):
        # text_feats=ckpt.checkpoint(self.clip_model.encode_text,text)
        text_feats=self.clip_model.encode_text(text)
        text_feats=text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        text_feats=self.bn_neck_text(text_feats)
        return text_feats
    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            losses.update(reid_loss)
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v
                # alignment, many-to-many matching -> similar to supervised contrastive loss 
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_i=self.alignment_loss(box_embs,text_embs,assign_ids,text_pids)
                align_loss_t=self.alignment_loss(text_embs,box_embs,text_pids,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.5
                losses["loss_align_t"]=align_loss_t*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            box_embs=F.normalize(box_embs,dim=-1)
            # nms
            reid_feats = torch.split(
                box_embs, [len(instances_i) for instances_i in pred_instances]
            )
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh
            for pred_i, gt_i, reid_feats_i in zip(
                pred_instances, gt_instances, reid_feats
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
                pred_i.reid_feats = reid_feats_i[filter_mask][keep]
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
                # back to org scale
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)
            return pred_instances

    def forward_query(self, image_list, gt_instances):
        # one sentece for each query
        text_tokens=[]
        for inst in gt_instances:
            text_tokens.append(inst.descriptions[0][0])
        text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
        text_embs = F.normalize(text_embs, dim=-1)
        text_embs = torch.split(text_embs, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=text_embs[i])
            for i in range(len(text_embs))
        ]

@META_ARCH_REGISTRY.register()
class OimClipSplit(OimClip):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)

        # res5 for reid
        for p in self.clip_model.visual.layer4.parameters():
            p.requires_grad=True
    @classmethod
    def from_config(cls, cfg):
        ret = super(OimClip,cls).from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":32} # for det
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4) # with stride
        ret["roi_heads"] = ClipRes5ROIHeadsPsDetOnly(cfg, res5,res_output_shape)
        stride_block=clip_model.visual.layer4[0] # without stride
        stride_block.avgpool=nn.Identity()
        stride_block.downsample[0]=nn.Identity()
        ret["id_assigner"] = build_id_assigner(cfg)
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
        oim_text=copy.deepcopy(ret["oim_loss"])
        del oim_text.ulb_layer
        oim_text.ulb_layer=None
        ret["oim_loss_text"]=oim_text
        return ret
    def img_embes(self,roi_feats):
        roi_feats=self.clip_model.visual.layer4(roi_feats)
        return self.clip_model.visual.attnpool(roi_feats)
@META_ARCH_REGISTRY.register()
class OimClipSplitT2I(OimClipSplit):
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            losses.update(reid_loss)
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v
                # alignment, many-to-many matching -> similar to supervised contrastive loss 
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,box_embs,text_pids,assign_ids)
                losses["loss_align_t"]=align_loss_t
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)
@META_ARCH_REGISTRY.register()
class OimClipStride(OimClip):
    @classmethod
    def from_config(cls, cfg):
        ret = super(OimClip,cls).from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":32}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        roi_feat_spatial=[roi_feat_spatial[0]//2,roi_feat_spatial[1]//2]
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4) # with stride
        ret["roi_heads"] = ClipRes5ROIHeadsPs(cfg, res5,res_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
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
        oim_text=copy.deepcopy(ret["oim_loss"])
        del oim_text.ulb_layer
        oim_text.ulb_layer=None
        ret["oim_loss_text"]=oim_text
        return ret
@META_ARCH_REGISTRY.register()
class OimClipT2I(OimClip):
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            losses.update(reid_loss)
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v
                # alignment, many-to-many matching -> similar to supervised contrastive loss 
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,box_embs,text_pids,assign_ids)
                losses["loss_align_t"]=align_loss_t
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimple(OimClip):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss=ProtoConLoss()
    @classmethod
    def from_config(cls, cfg):
        ret = super(OimClip,cls).from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":32}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        roi_feat_spatial=[roi_feat_spatial[0]//2,roi_feat_spatial[1]//2]
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4) # with stride
        ret["roi_heads"] = ClipRes5ROIHeadsPs(cfg, res5,res_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
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
        ret["oim_loss_text"]=None
        return ret

    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            losses.update(reid_loss)
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                # alignment, text-to-image matching
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs=F.normalize(v_features,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
                losses["loss_align_t"]=align_loss_t
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipDetOnly(OimClipSimple):
    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances
        )
        losses.update(proposal_losses)
        if self.training:
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

@META_ARCH_REGISTRY.register()
class OimClipSimpleDC2(OimClipSimple):
    @classmethod
    def from_config(cls, cfg):
        ret =SearchBaseTBPS.from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":16}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4) # without stride
        stride_block=res5[0]
        stride_block.avgpool=nn.Identity()
        stride_block.downsample[0]=nn.Identity()
        ret["roi_heads"] = ClipRes5ROIHeadsPs(cfg, res5,res_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
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
        ret["oim_loss_text"]=None
        return ret

@META_ARCH_REGISTRY.register()
class OimClipSimpleDC2Bi(OimClipSimple):
    @classmethod
    def from_config(cls, cfg):
        ret = SearchBaseTBPS.from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":16}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4) # without stride
        stride_block=res5[0]
        stride_block.avgpool=nn.Identity()
        stride_block.downsample[0]=nn.Identity()
        ret["roi_heads"] = ClipRes5ROIHeadsPs(cfg, res5,res_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
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
        oim_text=copy.deepcopy(ret["oim_loss"])
        del oim_text.ulb_layer
        oim_text.ulb_layer=None
        ret["oim_loss_text"]=oim_text
        return ret
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs=F.normalize(v_features,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
                losses["loss_align_t"]=align_loss_t*0.5
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                align_loss_i=self.alignment_loss(box_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleBi(OimClipSimpleDC2Bi):
    @classmethod
    def from_config(cls, cfg):
        ret = SearchBaseTBPS.from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":32}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        roi_feat_spatial=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        roi_feat_spatial=[roi_feat_spatial[0]//2,roi_feat_spatial[1]//2]
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            cls_pos=pos[0]
            spatial_size=round(pos[1:].shape[0]**0.5)
            img_pos=pos[1:].reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos[None], size=roi_feat_spatial, mode='bicubic', align_corners=False)[0]
            img_pos=img_pos.permute(1,2,0)
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(torch.cat([cls_pos[None],img_pos.flatten(0,1)]))
        if cfg.PERSON_SEARCH.DET.CLIP.FROZEN_BN:
            clip_model.visual=convert_frozen_bn(clip_model.visual)
        ret["clip_model"]=clip_model
        ret["freeze_at_stage2"]=cfg.PERSON_SEARCH.DET.CLIP.FREEZE_AT_STAGE2
        ret["proposal_generator"] = build_proposal_generator(cfg, res_output_shape)
        res5=copy.deepcopy(clip_model.visual.layer4) # without stride
        ret["roi_heads"] = ClipRes5ROIHeadsPs(cfg, res5,res_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
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
        oim_text=copy.deepcopy(ret["oim_loss"])
        del oim_text.ulb_layer
        oim_text.ulb_layer=None
        ret["oim_loss_text"]=oim_text
        return ret



from psd2.utils.simple_tokenizer import SimpleTokenizer
import random
import math
@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLM(OimClipSimpleBi):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_attn = nn.MultiheadAttention(embed_dim,
                                                embed_dim // 64)
        self.cross_modal_transformer = Transformer(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_pre_t = nn.LayerNorm(embed_dim)
        self.ln_pre_i = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        num_transp=0
        for n,p in self.cross_modal_transformer.named_parameters():
            num_transp+=p.view(-1).shape[0]
        for p in self.cross_attn.parameters():
            num_transp+=p.view(-1).shape[0]
        
        print("num_parameters_transformer:"+str(num_transp))
    def random_masked_tokens_and_labels(self,all_tokens):
        # random_masked tokens and labels
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        all_masked_tokens=[]
        all_labels=[]
        for tokens in all_tokens:
            if isinstance(tokens,torch.Tensor):
                tokens=tokens.clone().cpu().numpy()
            labels = []
            for i, token in enumerate(tokens):
                if 0 < token < 49405:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        prob /= 0.15

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = mask

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.choice(token_range)

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        labels.append(token)
                    else:
                        # no masking token (will be ignored by loss function later)
                        labels.append(0)
                else:
                    labels.append(0)
            
            if all(l == 0 for l in labels):
                # at least mask 1
                labels[1] = tokens[1]
                tokens[1] = mask
            all_masked_tokens.append(torch.tensor(tokens))
            all_labels.append(torch.tensor(labels))

        return torch.stack(all_masked_tokens).to(self.device),torch.stack(all_labels).to(self.device)
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    def mlm_loss(self,i_features,tokens):
        masked_tokens,token_labels=self.random_masked_tokens_and_labels(tokens)
        mlm_feats =ckpt.checkpoint(self.clip_model.encode_text,masked_tokens)
        x =ckpt.checkpoint(self.cross_former,mlm_feats, i_features, i_features)

        x =ckpt.checkpoint(self.mlm_head,x)  # [batch_size, text_len, num_colors]

        scores = x.float().reshape(-1, self.clip_model.vocab_size)
        return {"loss_mlm":F.cross_entropy(scores,token_labels.reshape(-1),ignore_index=0) *0.1}
    def img_embes(self,roi_feats):
        def inner_forward(x):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.clip_model.visual.attnpool.num_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False
            )
            return x
        img_feats=inner_forward(roi_feats)
        embs,feats=img_feats[0],img_feats[1:]
        if self.training:
            return embs,feats
        else:
            return embs
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs=F.normalize(v_features,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
                losses["loss_align_t"]=align_loss_t*0.5
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                align_loss_i=self.alignment_loss(box_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)


@META_ARCH_REGISTRY.register()
class OimClipSimpleBiVisionLanguageGTInfer(OimClipSimpleBi):
    def forward_gallery(self, image_list, gt_instances):
        assert not self.training
        pids=torch.cat([inst.gt_pids for inst in gt_instances])
        text_tokens=[]
        for inst in gt_instances:
            text_tokens.extend([des[0] for des in inst.descriptions])
        text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
        text_embs = F.normalize(text_embs, dim=-1)
        
        features = self.backbone(image_list.tensor)
        boxes=[inst.gt_boxes for inst in gt_instances]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], boxes
        )
        del features
        box_embs = self.img_embes(box_features)
        box_embs = self.bn_neck(box_embs)
        box_embs=F.normalize(box_embs,dim=-1)
        pos_mask=pids>0
        return pids[pos_mask],text_embs[pos_mask],box_embs[pos_mask]

@META_ARCH_REGISTRY.register()
class OimClipSimpleOnlineBiMLM(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss=SupConLoss()
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                v_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids,assign_ids)
                losses["loss_align_t"]=align_loss_t*0.5

                pos_mask=assign_ids>-1
                assign_ids=assign_ids[pos_mask]
                v_embs = v_embs[pos_mask]
                align_loss_i=self.alignment_loss(v_embs,text_embs,assign_ids,text_pids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)


@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLM2(OimClipSimpleBiMLM):
    def img_embes(self,roi_feats):
        def inner_forward(x):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.clip_model.visual.attnpool.num_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False
            )
            return x
        img_feats=ckpt.checkpoint(inner_forward,roi_feats)
        embs,feats=img_feats[0],img_feats[1:]
        if self.training:
            return embs,feats
        else:
            return embs
    def text_embeds(self,text):
        text_feats=ckpt.checkpoint(self.clip_model.encode_text,text)
        # text_feats=self.clip_model.encode_text(text,ckpt=True)
        text_feats=text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        text_feats=self.bn_neck_text(text_feats)
        return text_feats
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        def inner_forward(q,k,v):
            q=q.permute(1, 0, 2)
            k=k.permute(1, 0, 2)
            v=v.permute(1, 0, 2)
            x = self.cross_attn(
                        self.ln_pre_t(q),
                        self.ln_pre_i(k),
                        self.ln_pre_i(v),
                        need_weights=False)[0]
            return x
        x=ckpt.checkpoint(inner_forward,q,k,v)
        x = self.cross_modal_transformer(x,ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    # sample at most 3 roi feats for each txt token
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc*2:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc*2/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc*2]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens*2)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs=F.normalize(v_features,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
                losses["loss_align_t"]=align_loss_t*0.5
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                align_loss_i=self.alignment_loss(box_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)


#NOTE model parallel, the outer training script calls model.to(device)
from collections import namedtuple
class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__
@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLM3(OimClipSimpleBiMLM):
    
    def load_state_dict(self, *args, **kws):
        state_dict = args[0]
        is_resume=False
        for np in state_dict:
            if "cross_modal_transformer" in np:
                is_resume=True
        if is_resume:
            rst=super().load_state_dict(*args, **kws)
        else:
            rst=_IncompatibleKeys([],[])
        self.cross_modal_transformer=self.cross_modal_transformer.to("cuda:1")
        self.cross_attn=self.cross_attn.to("cuda:1")
        self.ln_pre_t=self.ln_pre_t.to("cuda:1")
        self.ln_pre_i=self.ln_pre_i.to("cuda:1")
        self.ln_post=self.ln_post.to("cuda:1")
        self.mlm_head=self.mlm_head.to("cuda:1")
        return rst

    def img_embes(self,roi_feats):
        def inner_forward(x):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.clip_model.visual.attnpool.num_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False
            )
            return x
        img_feats=ckpt.checkpoint(inner_forward,roi_feats)
        embs,feats=img_feats[0],img_feats[1:]
        if self.training:
            return embs,feats
        else:
            return embs
    def text_embeds(self,text):
        text_feats=ckpt.checkpoint(self.clip_model.encode_text,text)
        # text_feats=self.clip_model.encode_text(text,ckpt=True)
        text_feats=text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        text_feats=self.bn_neck_text(text_feats)
        return text_feats
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        def inner_forward(q,k,v):
            q=q.permute(1, 0, 2)
            k=k.permute(1, 0, 2)
            v=v.permute(1, 0, 2)
            x = self.cross_attn(
                        self.ln_pre_t(q),
                        self.ln_pre_i(k),
                        self.ln_pre_i(v),
                        need_weights=False)[0]
            return x
        x=ckpt.checkpoint(inner_forward,q,k,v)
        x = self.cross_modal_transformer(x,ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    def mlm_loss(self,i_features,tokens):
        masked_tokens,token_labels=self.random_masked_tokens_and_labels(tokens)
        mlm_feats =self.clip_model.encode_text(masked_tokens,ckpt=True)
        mlm_feats=mlm_feats.to("cuda:1")
        i_features=i_features.to("cuda:1")
        x =self.cross_former(mlm_feats, i_features, i_features)

        x =ckpt.checkpoint(self.mlm_head,x)  # [batch_size, text_len, num_colors]

        scores = x.float().reshape(-1, self.clip_model.vocab_size).to("cuda:0")
        return {"loss_mlm":F.cross_entropy(scores,token_labels.reshape(-1),ignore_index=0)*0.5}
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one token for one roi feature
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_ifeats=id_feats.shape[0]
                    num_desc=len(p_tokens)
                    sampled_idxs=torch.randint(0,num_desc,(num_ifeats,)).tolist()
                    img_features.append(id_feats)
                    text_tokens.extend([p_tokens[i] for i in sampled_idxs])
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs=F.normalize(v_features,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
                losses["loss_align_t"]=align_loss_t*0.5
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                align_loss_i=self.alignment_loss(box_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)
    
@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMD(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMLM,self).__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = CrossAttentionTransformer(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,k,v,with_ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDFully(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMLM,self).__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = FullyCrossAttentionTransformer(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        num_transp=0
        for p in self.cross_modal_transformer.parameters():
            num_transp+=p.view(-1).shape[0]
        print("num_parameters_transformer:"+str(num_transp))
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,k,v,with_ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMIMDFully(OimClipSimpleBiMLMDFully):
    @configurable
    def __init__(
        self,
        mim_ratio,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMIMDFully,self).__init__(*args, **kws)
        self.mim_ratio=mim_ratio
        embed_dim=self.clip_model.text_projection.shape[1]
        self.mim_token=nn.Parameter(torch.randn(1, embed_dim)/ embed_dim ** 0.5)
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["mim_ratio"]=cfg.PERSON_SEARCH.REID.MIM_RATIO
        return ret
    def random_masked_tokens_and_labels(self,all_tokens):
        
        all_masked_tokens=[]
        for tokens in all_tokens:
            num_tokens=tokens.shape[0]
            prob = torch.rand(num_tokens,device=all_tokens.device)
            re_idxs=torch.arange(0,num_tokens,dtype=torch.long,device=all_tokens.device)
            re_idxs[prob<self.mim_ratio]=-1
            cat_tokens=torch.cat([tokens,self.mim_token],dim=0)
            masked_tokens=cat_tokens[re_idxs]
            all_masked_tokens.append(masked_tokens)
        return torch.stack(all_masked_tokens)
    def mlm_loss(self,i_features,tokens):
        masked_i_features=self.random_masked_tokens_and_labels(i_features)
        text_feats =ckpt.checkpoint(self.clip_model.encode_text,torch.stack(tokens).to(self.device))
        rec_i_features =ckpt.checkpoint(self.cross_former,masked_i_features, text_feats, text_feats)
        num_i_tokens=i_features.shape[0]*i_features.shape[1]
        tgt_i_features=i_features.detach()
        cos_dist=1-(F.normalize(rec_i_features,dim=-1)*F.normalize(tgt_i_features,dim=-1)).sum(-1)
        return {"loss_mim":cos_dist.sum()/num_i_tokens}


@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMHMDFully(OimClipSimpleBiMIMDFully):
    def random_masked_tokens_and_labels(self,all_tokens):
        return super(OimClipSimpleBiMIMDFully,self).random_masked_tokens_and_labels(all_tokens)
    def mlm_loss(self,i_features,tokens):
        return super(OimClipSimpleBiMIMDFully,self).mlm_loss(i_features,tokens)
    def random_masked_patches(self,patch_tokens):
        all_masked_tokens=[]
        for tokens in patch_tokens:
            num_tokens=tokens.shape[0]
            prob = torch.rand(num_tokens,device=patch_tokens.device)
            re_idxs=torch.arange(0,num_tokens,dtype=torch.long,device=patch_tokens.device)
            re_idxs[prob<self.mim_ratio]=-1
            cat_tokens=torch.cat([tokens,self.mim_token],dim=0)
            masked_tokens=cat_tokens[re_idxs]
            all_masked_tokens.append(masked_tokens)
        return torch.stack(all_masked_tokens)
    def mim_loss(self,i_features,t_features):
        masked_i_features=self.random_masked_patches(i_features)
        rec_i_features =ckpt.checkpoint(self.cross_former,masked_i_features, t_features,t_features)
        num_i_tokens=i_features.shape[0]*i_features.shape[1]
        tgt_i_features=i_features.detach()
        cos_dist=1-(F.normalize(rec_i_features,dim=-1)*F.normalize(tgt_i_features,dim=-1)).sum(-1)
        return {"loss_mim":cos_dist.sum()/num_i_tokens}
    
    def text_embeds_features(self,text):
        text_feats=self.clip_model.encode_text(text)
        text_embes=text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        text_embes=self.bn_neck_text(text_embes)
        return text_feats,text_embes
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    if pid>-1:
                        id_feats=img_rois[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_feats.shape[0]<num_desc:
                            id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                        img_features.append(id_feats[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim & mim
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
                losses["loss_mim"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_features,text_embs=self.text_embeds_features(torch.stack(text_tokens).to(self.device))
                losses.update(self.mim_loss(roi_feats,text_features))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs=F.normalize(v_features,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
                losses["loss_align_t"]=align_loss_t*0.5
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                align_loss_i=self.alignment_loss(box_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMHM2DFully(OimClipSimpleBiMHMDFully):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMHM2DFully,self).__init__(*args, **kws)
        self.cross_modal_transformer_mim=copy.deepcopy(self.cross_modal_transformer)
    def cross_former_mim(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer_mim(q,k,v,with_ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    def mim_loss(self,i_features,t_features):
        masked_i_features=self.random_masked_patches(i_features)
        rec_i_features =ckpt.checkpoint(self.cross_former_mim,masked_i_features, t_features,t_features)
        num_i_tokens=i_features.shape[0]*i_features.shape[1]
        tgt_i_features=i_features.detach()
        cos_dist=1-(F.normalize(rec_i_features,dim=-1)*F.normalize(tgt_i_features,dim=-1)).sum(-1)
        return {"loss_mim":cos_dist.sum()/num_i_tokens}

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMIML2DFully(OimClipSimpleBiMIMDFully):

    def mlm_loss(self,i_features,tokens):
        masked_i_features=self.random_masked_tokens_and_labels(i_features)
        text_feats =ckpt.checkpoint(self.clip_model.encode_text,torch.stack(tokens).to(self.device))
        rec_i_features =ckpt.checkpoint(self.cross_former,masked_i_features, text_feats, text_feats)
        num_i_tokens=i_features.shape[0]*i_features.shape[1]
        tgt_i_features=i_features.detach()
        l2_dist=F.mse_loss(rec_i_features,tgt_i_features,reduction="none")
        return {"loss_mim":l2_dist.sum()/num_i_tokens*0.001}
           
@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMHML2DFully(OimClipSimpleBiMHMDFully):
    def mim_loss(self,i_features,t_features):
        masked_i_features=self.random_masked_patches(i_features)
        rec_i_features =ckpt.checkpoint(self.cross_former,masked_i_features, t_features,t_features)
        num_i_tokens=i_features.shape[0]*i_features.shape[1]
        tgt_i_features=i_features.detach()
        l2_dist=F.mse_loss(rec_i_features,tgt_i_features,reduction="none")
        return {"loss_mim":l2_dist.sum()/num_i_tokens*0.001}

from psd2.utils.simple_tokenizer import SimpleTokenizer
import cv2
import torchvision.transforms.functional as tvF
from psd2.utils.visualizer import Visualizer
@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDFullyVis(OimClipSimpleBiMLMDFully):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
    def img_embes(self,roi_feats):
        def inner_forward(x):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.clip_model.visual.attnpool.num_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False
            )
            return x
        img_feats=inner_forward(roi_feats)
        embs,feats=img_feats[0],img_feats[1:]
        return embs,feats
    def forward_gallery(self, image_list, gt_instances):
        pids=torch.cat([inst.gt_pids for inst in gt_instances])
        text_tokens=[]
        for inst in gt_instances:
            text_tokens.extend([des[0] for des in inst.descriptions])
        pos_text_tokens=[]
        for t,pid in zip(text_tokens,pids.cpu().numpy()):
            if pid>-1:
                pos_text_tokens.append(t)
        text_tokens=pos_text_tokens
        pos_mask=pids>-1
        masked_tokens,token_labels=self.random_masked_tokens_and_labels(text_tokens)
        text_masked=[]
        text_gts=[]
        for i in range(len(text_tokens)):
            text_gts.append(self.tokenizer.decode(text_tokens[i].cpu().numpy()[1:]))
            text_masked.append(self.tokenizer.decode(masked_tokens[i].cpu().numpy()[1:]))
        mlm_feats=self.clip_model.encode_text(masked_tokens)
        features = self.backbone(image_list.tensor)
        boxes=[inst.gt_boxes for inst in gt_instances]
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], boxes
        )
        box_features=box_features[pos_mask]
        _,box_feats = self.img_embes(box_features)
        inter_attns=self.mlm_with_attn(box_feats,mlm_feats)
        tgt_size=(384,192)
        crops=[]
        for bi,boxesi in enumerate(boxes):
            imgi=image_list.tensor[bi]
            h,w=imgi.shape[-2:]
            xmin, ymin, xmax, ymax = boxesi.tensor.unbind(1)
            xmin = xmin.clamp(0,w-1).int()
            xmax = xmax.clamp(1,w).ceil().int()
            ymin = ymin.clamp(0,h-1).int()
            ymax = ymax.clamp(1,h).ceil().int()
            cropsi = []
            for x1,y1,x2,y2 in zip(xmin,ymin,xmax,ymax):
                crop_0=imgi[:,y1:y2,x1:x2].clone() # scale org
                crop_0=torch.nn.functional.interpolate(crop_0[None], size=tgt_size, mode='bilinear', align_corners=False)[0]
                crop_0=crop_0*self.pixel_std+self.pixel_mean
                cropsi.append(crop_0)
            crops.extend(cropsi)
        crops=torch.stack(crops)
        crops=crops[pos_mask]
        self.vis_mlm(crops,text_gts,text_masked,torch.stack(inter_attns),token_labels)
        
    def mlm_with_attn(self,roi_feats,mlm_feats):
        inter_attns=[]
        q=mlm_feats
        k=v=roi_feats
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        for blk in self.cross_modal_transformer.resblocks:
            t,attn=blk.attn(blk.ln_1(q),k,v,need_weights=True)
            inter_attns.append(F.softmax(attn,dim=-1))
            q=q+t
            q=q+blk.mlp(blk.ln_2(q))
        return inter_attns
    def vis_mlm(self,crops,text_org,text_masked,token_attn,mask_labels):
        storage=get_event_storage()
        for i in range(crops.shape[0]):
            img=crops[i]
            mask_idxs=mask_labels[i].nonzero(as_tuple=True)[0]
            mask_token_attn=token_attn[:,i][:,mask_idxs] #M * L * S
            mask_token_attn=mask_token_attn.reshape(mask_token_attn.shape[0],-1,8,4)
            inter_attns=torch.nn.functional.interpolate(mask_token_attn, size=crops[i].shape[-2:], mode='bilinear', align_corners=False) # M * L * H * W
            nl,nm=inter_attns.shape[:2]
            inter_attns=inter_attns.flatten(0,1)
            attn_imgs=[]
            for attn in inter_attns:
                attn_img=255*(attn-attn.min())/(attn.max() - attn.min() + 1e-12)
                attn_img=attn_img*attn_img
                attn_img=255*(attn-attn.min())/(attn.max() - attn.min() + 1e-12)
                attn_img = attn_img.cpu().numpy().astype(np.uint8)
                attn_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_JET)  # bgr hwc
                attn_img = torch.tensor(
                    attn_img[..., ::-1].copy(), device=img.device, dtype=img.dtype
                ).permute(2, 0, 1)/255
                coeff = 0.3
                attn_img = (1 - coeff) * img + coeff * attn_img
                attn_imgs.append(attn_img)
            cat_attn_imgs=torch.stack(attn_imgs)
            cat_attn_imgs=cat_attn_imgs.reshape(nl,nm,cat_attn_imgs.shape[-3],cat_attn_imgs.shape[-2],cat_attn_imgs.shape[-1]) # M * L * C * H * W
            cat_attn_imgs=cat_attn_imgs.permute(2,1,3,0,4).contiguous()
            cat_attn_imgs=cat_attn_imgs.flatten(3,4).flatten(1,2) # C * LH * MW
            storage.put_image("samp{}/attn".format(i),cat_attn_imgs)
            text_img=torch.zeros((40,2048,3)).cpu().numpy()+255
            visualize_text = Visualizer(text_img)
            visualize_text.draw_text(
                    text_masked[i],
                    [0,0],
                    horizontal_alignment="left",
                    vertical_alignment="top",
                    color="black",
                    bg_color="white",
                )
            visualize_text.draw_text(
                    text_org[i],
                    [0,39],
                    horizontal_alignment="left",
                    vertical_alignment="bottom",
                    color="black",
                    bg_color="white",
                )
            text_img=visualize_text.get_output().get_image()
            text_img=tvF.to_tensor(text_img)
            storage.put_image("samp{}/text".format(i),text_img)



        

@META_ARCH_REGISTRY.register()
class OimClipSimpleT2IMLMDFully(OimClipSimpleBiMLM):
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs=F.normalize(v_features,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
                losses["loss_align_t"]=align_loss_t # *0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)
@META_ARCH_REGISTRY.register()
class OimClipSimpleI2TMLMDFully(OimClipSimpleBiMLM):
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                pos_mask=assign_ids>-1
                box_embs=box_embs[pos_mask]
                assign_ids=assign_ids[pos_mask]
                box_embs=F.normalize(box_embs,dim=-1)
                align_loss_i=self.alignment_loss(box_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)
    


@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDFully2(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMLM,self).__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = FullyCrossAttentionTransformer(width=embed_dim,
                                                    layers=2,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        num_transp=0
        for p in self.cross_modal_transformer.parameters():
            num_transp+=p.view(-1).shape[0]
        print("num_parameters_transformer:"+str(num_transp))
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,k,v,with_ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDFully6(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMLM,self).__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = FullyCrossAttentionTransformer(width=embed_dim,
                                                    layers=6,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        num_transp=0
        for p in self.cross_modal_transformer.parameters():
            num_transp+=p.view(-1).shape[0]
        print("num_parameters_transformer:"+str(num_transp))
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,k,v,with_ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

@META_ARCH_REGISTRY.register()
class OimClipSimpleOnlineBiMLMDFully(OimClipSimpleBiMLMDFully):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss=SupConLoss()
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                v_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids,assign_ids)
                losses["loss_align_t"]=align_loss_t*0.5

                pos_mask=assign_ids>-1
                assign_ids=assign_ids[pos_mask]
                v_embs = v_embs[pos_mask]
                align_loss_i=self.alignment_loss(v_embs,text_embs,assign_ids,text_pids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDUni(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMLM,self).__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = UniCrossAttentionTransformer(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        del v
        q=q.permute(1, 0, 2)
        kv=k.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,kv,with_ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDExt(OimClipSimpleBiMLMD):
    def img_embes(self,roi_feats):
        def inner_forward(x):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.clip_model.visual.attnpool.num_heads,
                q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
                k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
                v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias, self.clip_model.visual.attnpool.k_proj.bias, self.clip_model.visual.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False
            )
            return x
        img_feats=inner_forward(roi_feats)
        embs=img_feats[0]
        if self.training:
            return embs,img_feats
        else:
            return embs

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMD6(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMLM,self).__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = CrossAttentionTransformer(width=embed_dim,
                                                    layers=6,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,k,v,with_ckpt=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDPostNorm(OimClipSimpleBiMLMD):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleBiMLM,self).__init__(*args, **kws)
        self.tokenizer=SimpleTokenizer()
        embed_dim=self.clip_model.text_projection.shape[1]
        self.cross_modal_transformer = CrossAttentionTransformerPostNorm(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_post = nn.Identity() # nn.LayerNorm(embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(embed_dim, embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', nn.LayerNorm(embed_dim)),
                            ('fc', nn.Linear(embed_dim, self.clip_model.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)


@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMDPos(OimClipSimpleBiMLMD):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        embed_dim=self.clip_model.text_projection.shape[1]
        positional_embedding_dt = nn.Parameter(torch.empty(self.clip_model.context_length, embed_dim))
        il=self.clip_model.visual.attnpool.positional_embedding.shape[0]
        positional_embedding_di=nn.Parameter(torch.randn((il-1), embed_dim)/ embed_dim ** 0.5)
        self.cross_modal_transformer.pos_q=positional_embedding_dt
        self.cross_modal_transformer.pos_k=positional_embedding_di
    

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLMHD(OimClipSimpleBiMLMD):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.mlm_head_h=nn.ModuleList([copy.deepcopy(self.mlm_head) for _ in range(3)])
        self.ln_post_h=nn.ModuleList([copy.deepcopy(self.ln_post) for _ in range(3)])

    def cross_former(self, q, k, v):
        # inputs are NLD
        # NLD -> LND
        q=q.permute(1, 0, 2)
        k=k.permute(1, 0, 2)
        v=v.permute(1, 0, 2)
        x = self.cross_modal_transformer(q,k,v,with_ckpt=True,return_inter=True)
        h_out=[]
        for i,inter in enumerate(x[:-1]):
            inter = inter.permute(1, 0, 2)  # LND -> NLD
            h_out.append(self.ln_post_h[i](inter))
        h_out.append(self.ln_post(x[-1]))
        return h_out
    def mlm_loss(self,i_features,tokens):
        masked_tokens,token_labels=self.random_masked_tokens_and_labels(tokens)
        mlm_feats =self.clip_model.encode_text(masked_tokens,ckpt=True)
        h_x =self.cross_former(mlm_feats, i_features, i_features)
        h_loss=0
        for i,x in enumerate(h_x[:-1]):
            x =ckpt.checkpoint(self.mlm_head_h[i],x)  # [batch_size, text_len, num_colors]
            scores = x.float().reshape(-1, self.clip_model.vocab_size)
            h_loss+=F.cross_entropy(scores,token_labels.reshape(-1),ignore_index=0)
        x =ckpt.checkpoint(self.mlm_head,h_x[-1])
        scores = x.float().reshape(-1, self.clip_model.vocab_size)
        h_loss+=F.cross_entropy(scores,token_labels.reshape(-1),ignore_index=0)
        return {"loss_mlm":h_loss/(len(self.mlm_head_h)+1)*0.5}

@META_ARCH_REGISTRY.register()
class OimClipSimpleBiMLME(OimClipSimpleBiMLM):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        del self.cross_attn
    def cross_former(self, x):
        # inputs are NLD
        # NLD -> LND
        x=x.permute(1, 0, 2)
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    def mlm_loss(self,i_features,tokens):
        masked_tokens,token_labels=self.random_masked_tokens_and_labels(tokens)
        mlm_feats =ckpt.checkpoint(self.clip_model.encode_text,masked_tokens)
        t_feats=self.ln_pre_t(mlm_feats)
        i_feats=self.ln_pre_i(i_features)
        x=torch.cat([t_feats,i_feats],dim=1)
        x =ckpt.checkpoint(self.cross_former,x)
        x=x[:,:t_feats.shape[1]]
        x =ckpt.checkpoint(self.mlm_head,x)  # [batch_size, text_len, num_colors]

        scores = x.float().reshape(-1, self.clip_model.vocab_size)
        return {"loss_mlm":F.cross_entropy(scores,token_labels.reshape(-1),ignore_index=0)*0.5}


@META_ARCH_REGISTRY.register()
class OimClipSimpleOnlineBi(OimClipSimpleBi):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss=SupConLoss()
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                v_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids,assign_ids)
                losses["loss_align_t"]=align_loss_t*0.5

                pos_mask=assign_ids>-1
                assign_ids=assign_ids[pos_mask]
                v_embs = v_embs[pos_mask]
                align_loss_i=self.alignment_loss(v_embs,text_embs,assign_ids,text_pids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleOnlineBiMLMD(OimClipSimpleBiMLMD):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss=SupConLoss()
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]
            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                v_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids,assign_ids)
                losses["loss_align_t"]=align_loss_t*0.5

                pos_mask=assign_ids>-1
                assign_ids=assign_ids[pos_mask]
                v_embs = v_embs[pos_mask]
                align_loss_i=self.alignment_loss(v_embs,text_embs,assign_ids,text_pids)
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)


@META_ARCH_REGISTRY.register()
class OimClipSimpleCombineBi(OimClipSimpleBi):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss_online=SupConLoss()
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                v_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss_online(text_embs,v_embs,text_pids,assign_ids)
                losses["loss_align_t_online"]=align_loss_t*0.25
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs_off=F.normalize(v_features,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs_off,text_pids)
                losses["loss_align_t"]=align_loss_t*0.25

                pos_mask=assign_ids>-1
                assign_ids=assign_ids[pos_mask]
                v_embs = v_embs[pos_mask]
                align_loss_i=self.alignment_loss_online(v_embs,text_embs,assign_ids,text_pids)
                losses["loss_align_i_online"]=align_loss_i*0.25
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                align_loss_i=self.alignment_loss(v_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.25
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleCombineBiMLMD(OimClipSimpleBiMLMD):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss_online=SupConLoss()
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            losses.update(proposal_losses)
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]
            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs,box_feats  = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                if (gt_ids>-1).nonzero().shape[0]==0:
                    continue
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    # there can be multiple text seq for one id
                    id_feats=img_rois[roi_ids==pid]
                    num_desc=len(p_tokens)
                    if id_feats.shape[0]<num_desc:
                        id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                    sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                    img_features.append(id_feats[sampled_idxs])
                    text_tokens.extend(p_tokens)
            if len(img_features)==0:
                losses["loss_mlm"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                v_embs=F.normalize(box_embs,dim=-1)
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss_online(text_embs,v_embs,text_pids,assign_ids)
                losses["loss_align_t_online"]=align_loss_t*0.25
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs_off=F.normalize(v_features,dim=-1)
                align_loss_t=self.alignment_loss(text_embs,v_embs_off,text_pids)
                losses["loss_align_t"]=align_loss_t*0.25

                pos_mask=assign_ids>-1
                assign_ids=assign_ids[pos_mask]
                v_embs = v_embs[pos_mask]
                align_loss_i=self.alignment_loss_online(v_embs,text_embs,assign_ids,text_pids)
                losses["loss_align_i_online"]=align_loss_i*0.25
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs=F.normalize(lb_features,dim=-1)
                align_loss_i=self.alignment_loss(v_embs,t_embs,assign_ids)
                losses["loss_align_i"]=align_loss_i*0.25
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleHybridBi(OimClipSimpleBi):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.alignment_loss=SupConLoss()
    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
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
            assign_ids = torch.cat(assign_ids)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                losses["loss_align_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                # alignment, text-to-image matching
                v_embs_on=F.normalize(box_embs,dim=-1)
                lb_features=self.oim_loss.lb_layer.lookup_table
                ulb_features=self.oim_loss.ulb_layer.queue
                v_features=torch.cat([lb_features,ulb_features],dim=0)
                v_embs_off=F.normalize(v_features,dim=-1)
                assign_ids_off=torch.cat([
                    torch.arange(lb_features.shape[0],dtype=assign_ids.dtype,device=assign_ids.device),
                    torch.zeros(ulb_features.shape[0],dtype=assign_ids.dtype,device=assign_ids.device)-1])
                text_embs=F.normalize(text_embs,dim=-1)
                align_loss_t=self.alignment_loss(
                    text_embs,torch.cat([v_embs_on,v_embs_off],dim=0),
                    text_pids,torch.cat([assign_ids,assign_ids_off]))
                losses["loss_align_t"]=align_loss_t*0.5

                pos_mask=assign_ids>-1
                assign_ids=assign_ids[pos_mask]
                v_embs = v_embs_on[pos_mask]
                lb_features=self.oim_loss_text.lb_layer.lookup_table
                t_embs_off=F.normalize(lb_features,dim=-1)
                text_pids_off=torch.arange(lb_features.shape[0],dtype=text_pids.dtype,device=text_pids.device)
                align_loss_i=self.alignment_loss(
                    v_embs,torch.cat([text_embs,t_embs_off],dim=0),
                    assign_ids,torch.cat([text_pids,text_pids_off]))
                losses["loss_align_i"]=align_loss_i*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

class ClipRes5ROIHeadsPs(Res5ROIHeads):
    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return ckpt.checkpoint(self.res5,x) if self.training else self.res5(x)
    @classmethod
    def from_config(cls, cfg,res5, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["box_predictor"] = (
            FastRCNNOutputLayersPs(
                cfg, ShapeSpec(channels=2048, height=1, width=1)
            )
        )
        ret["res5"]=res5
        return ret
    @classmethod
    def _build_res5_block(cls, cfg):
        return nn.Identity(),2048 # trivial impl
    def _sample_proposals(
        self,
        matched_idxs: torch.Tensor,
        matched_labels: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        num_gts=gt_classes.numel()
        has_gt = num_gts > 0
        
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels_with_gt(
            gt_classes,
            self.batch_size_per_image,
            self.positive_fraction,
            self.num_classes,
            num_gts
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]
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
        # NOTE self.proposal_append_gt is True by default
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
            [features[f] for f in self.in_features], proposal_boxes
        )
        box_embs = box_features.mean(dim=[2, 3])
        predictions = self.box_predictor(box_embs)

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
                box_features,
                losses,
                pos_match_indices,
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, box_features, {}, None
class ClipRes5ROIHeadsPsDetOnly(ClipRes5ROIHeadsPs):
    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return x,self.res5(x)
    def forward(
        self,
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
        box_features_share,box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        box_embs = box_features.mean(dim=[2, 3])
        predictions = self.box_predictor(box_embs)

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
                box_features_share,
                losses,
                pos_match_indices,
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, box_features_share, {}, None

class ClipRes5SeqROIHeadsPs(ClipRes5ROIHeadsPs):
    @configurable
    def __init__(self,*args,**kws):
        super().__init__(*args,**kws)
        self.res5_det=copy.deepcopy(self.res5)
        self.box_predictor_det=copy.deepcopy(self.box_predictor)
    def _shared_roi_transform_det(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return ckpt.checkpoint(self.res5_det,x)
    def forward(
        self,
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

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform_det(
            [features[f] for f in self.in_features], proposal_boxes
        )
        box_embs = box_features.mean(dim=[2, 3])
        predictions = self.box_predictor_det(box_embs)

        if self.training:
            # det only head

            losses = self.box_predictor_det.losses(predictions, proposals)
            ls = list(losses.keys())
            for k in ls:
                losses[k + "_1st"] = losses.pop(k)
        with torch.no_grad():
            # for second stage
            pred_instances = self.box_predictor_det.inference_unms(
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
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        box_embs = box_features.mean(dim=[2, 3])
        predictions = self.box_predictor(box_embs)
        del features
        if self.training:
            losses2 = self.box_predictor.losses(predictions, proposals)
            losses.update(losses2)
            with torch.no_grad():
                # for vis and id_assign
                pred_instances = self.box_predictor.inference_unms(
                    predictions, proposals
                )
            return (
                pred_instances,
                box_features,
                losses,
                pos_match_indices
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, box_features, {}, None

from psd2.layers import nonzero_tuple
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
def subsample_labels_with_gt(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int,num_gts:int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive= nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    gt_indices=torch.arange(0,num_gts,device=positive.device)+positive.numel()-num_gts
    perm1=torch.randperm(positive.numel()-num_gts, device=positive.device)[:num_pos-num_gts]
    perm1=torch.cat([perm1,gt_indices])
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

def convert_frozen_bn(module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = FrozenBatchNorm2d(module.num_features,module.eps)
        module_output.weight=module.weight.data
        module_output.bias=module.bias.data
        module_output.running_mean=module.running_mean
        module_output.running_var=module.running_var
    for name, child in module.named_children():
        module_output.add_module(name, convert_frozen_bn(child))
    del module
    return module_output

class FastRCNNOutputLayersPs(FastRCNNOutputLayers):
    def inference_unms(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
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

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_0 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.self_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor):
        # self attn
        q = q + self.attention(self.ln_0(q))
        # cross attn
        q = q + self.attn(self.ln_1(q),k,v,need_weights=False, attn_mask=None)[0]
        q = q + self.mlp(self.ln_2(q))
        return q

class ResidualFullyCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.self_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor):
        # cross attn
        q = q + self.attn(self.ln_1(q),k,v,need_weights=False, attn_mask=None)[0]
        q = q + self.mlp(self.ln_2(q))
        return q

class ResidualCrossAttentionBlockPostNorm(ResidualCrossAttentionBlock):
    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor):
        # self attn
        q = q + self.attention(q)
        q=self.ln_0(q)
        # cross attn
        q = q + self.attn(q,k,v,need_weights=False, attn_mask=None)[0]
        q=self.ln_1(q)
        q = q + self.mlp(q)
        q=self.ln_2(q)
        return q

class CrossAttentionTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualCrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.pos_q=None
        self.pos_k=None

    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,with_ckpt=False,return_inter=False):
        inter_out=[]
        if self.pos_q is not None:
            q=q+self.pos_q.unsqueeze(1)
        if with_ckpt:
            # def inner_forward()
            for blk in self.resblocks:
                if self.pos_k is not None:
                    k=k+self.pos_k.unsqueeze(1)
                q=ckpt.checkpoint(blk,q,k,v)
                if return_inter:
                    inter_out.append(q)
        else:
            for blk in self.resblocks:
                if self.pos_k is not None:
                    k=k+self.pos_k.unsqueeze(1)
                q=blk(q,k,v)
                if return_inter:
                    inter_out.append(q)
        if return_inter:
            q=inter_out
        return q

class FullyCrossAttentionTransformer(CrossAttentionTransformer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super(CrossAttentionTransformer,self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualFullyCrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.pos_q=None
        self.pos_k=None

class UniCrossAttentionTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, q: torch.Tensor,kv: torch.Tensor,with_ckpt=False,return_inter=False):
        inter_out=[]
        ql=q.shape[0]
        if with_ckpt:
            # def inner_forward()
            for blk in self.resblocks:
                q=torch.cat([q,kv],dim=0)
                q=ckpt.checkpoint(blk,q)
                q=q[:ql]
                if return_inter:
                    inter_out.append(q)
        else:
            for blk in self.resblocks:
                q=torch.cat([q,kv],dim=0)
                q=blk(q)
                q=q[:ql]
                if return_inter:
                    inter_out.append(q)
        if return_inter:
            q=inter_out
        return q


class CrossAttentionTransformerPostNorm(CrossAttentionTransformer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super(CrossAttentionTransformer,self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualCrossAttentionBlockPostNorm(width, heads, attn_mask) for _ in range(layers)])


class SupConLoss(nn.Module):
    #NOTE simplified
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature,contrast_feature, anchor_labels,contrast_labels):
        anchor_labels=anchor_labels.view(-1,1)
        contrast_labels=contrast_labels.view(-1,1)
        mask = torch.eq(anchor_labels, contrast_labels.T).float()
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

class ProtoConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ProtoConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature,proto_feature, anchor_labels):
        anchor_labels=anchor_labels.view(-1)
        # compute logits
        anchor_logits = torch.div(
            torch.matmul(anchor_feature, proto_feature.T),
            self.temperature)
        loss=F.cross_entropy(anchor_logits,anchor_labels,reduction="none")
        # loss
        loss = (self.temperature / self.base_temperature) * loss
        loss = loss.mean()

        return loss