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
from psd2.modeling.extend.clip_model import build_CLIP_from_openai_pretrained
import copy
import torch.utils.checkpoint as ckpt

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
        x = stem(x)
        x =ckpt.checkpoint(visual_encoder.layer1,x)
        x =ckpt.checkpoint(visual_encoder.layer2,x)
        x =ckpt.checkpoint(visual_encoder.layer3,x)
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
    
class ClipRes5ROIHeadsPs(Res5ROIHeads):
    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return ckpt.checkpoint(self.res5,x)
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
from psd2.layers import nonzero_tuple
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