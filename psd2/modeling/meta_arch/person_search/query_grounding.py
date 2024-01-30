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
import psd2.utils.comm as comm

logger = logging.getLogger(__name__)

# GeneralizedRCNN as reference
@META_ARCH_REGISTRY.register()
class ClipQueryGroundingBaseline(SearchBaseTBPS):
    @configurable
    def __init__(
        self,
        clip_model,
        visual_pos,
        grounding_transformer,
        bbox_embed,
        conf_embed,
        oim_loss_text,
        bn_neck,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.clip_model=clip_model
        self.oim_loss_text=oim_loss_text
        self.bn_neck = bn_neck
        self.bn_neck_text=copy.deepcopy(bn_neck)
        self.grounding_trans=grounding_transformer
        self.bbox_embed=bbox_embed
        self.conf_embed=conf_embed
        self.query_embed = nn.Embedding(1, clip_model.text_projection.shape[1])
        self.visual_pos=visual_pos
        
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE)
        # resize clip pos embedding
        with torch.no_grad():
            pos=clip_model.visual.attnpool.positional_embedding
            img_pos=pos[1:]
            clip_model.visual.attnpool.positional_embedding=nn.Parameter(img_pos)
        ret["clip_model"]=clip_model
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        oim_text=OIMLoss(cfg)
        del oim_text.ulb_layer
        oim_text.ulb_layer=None
        ret["oim_loss_text"]=oim_text
        embed_dim=clip_model.text_projection.shape[1]
        ret["visual_pos"]=PositionEmbeddingSine(embed_dim//2,normalize=True)
        ret["grounding_transformer"]=cls._build_grounding_transformer(embed_dim)
        conf_embed,bbox_embed=cls._build_out_embed(embed_dim)
        ret["conf_embed"]=conf_embed
        ret["bbox_embed"]=bbox_embed
        return ret
    @classmethod
    def _build_grounding_transformer(cls,embed_dim):
        trans=FullyCrossAttentionTransformer(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        scale = trans.width**-0.5

        proj_std = scale * ((2 * trans.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * trans.width)**-0.5
        for block in trans.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        return trans
    @classmethod
    def _build_out_embed(cls,embed_dim):
        # NOTE from DETR
        conf_embed = nn.Linear(embed_dim, 2)
        bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        return conf_embed,bbox_embed
    
    def m_pred_loss(self,m_query_conf,m_query_bbox,gt_instances,match_indices):
        losses={}
        num_pos=sum([(inst.gt_pids>-1).sum() for inst in gt_instances])
        
        if comm.get_world_size()>1:
            all_num_pos=comm.all_gather(num_pos)
            num_pos = sum([num.to("cpu") for num in all_num_pos])
            num_pos = torch.clamp(num_pos / comm.get_world_size(), min=1).item()
        else:
            num_pos=num_pos.item()
        for i in range(len(self.grounding_trans.resblocks)):
            ilosses=self.conf_loss(m_query_conf[i],gt_instances,match_indices,num_pos)
            ilosses.update(self.bbox_loss(m_query_bbox[i],gt_instances,match_indices,num_pos))
            for k,v in ilosses.items():
                losses[k+"_{}".format(i)]=v
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx,src_idx = torch.split(indices,[1,1],dim=1)
        return batch_idx.squeeze(1).cpu().tolist(),src_idx.squeeze(1).cpu().tolist()

    def conf_estimate(self,m_query_out,query_input):
        return self.conf_embed(m_query_out)
    def conf_loss(self,scores,gt_instances,match_indices,num_inst):
        bidx,gidx=self._get_src_permutation_idx(match_indices)
        gidx=[0]*len(gidx)
        scores=scores.permute(1,0,2).flatten(0,1)
        targets=torch.ones_like(scores,dtype=torch.long)
        targets[bidx,gidx]=0
        loss= F.cross_entropy(scores,targets[:,0],reduction='none')
        return {"loss_conf": loss.sum()/num_inst * 1.0}
    
    def bbox_loss(self,pred_boxes,gt_instances,match_indices,num_inst):
        # NOTE predictions are cchw in (0,1)
        pred_boxes=pred_boxes.permute(1,0,2).flatten(0,1)
        bidx,gidx=self._get_src_permutation_idx(match_indices)
        src_boxes = pred_boxes[bidx]
        cchw_gt_boxes=torch.cat([inst.gt_boxes.convert_mode(BoxMode.CCWH_REL,inst.image_size).tensor for inst in gt_instances])
        target_boxes = cchw_gt_boxes[gidx]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_inst * 5.0

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_inst *2.0
        return losses
        
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
        x =ckpt.checkpoint(visual_encoder.layer4,x) if self.training else visual_encoder.layer4(x)
        return x
    def img_embes(self,x):
        # NOTE remove cls token, dynamically interpolate pos
        def inner_forward(x):
            pos=self.clip_model.visual.attnpool.positional_embedding.clone()
            spatial_size=round(pos.shape[0]**0.5)
            img_pos=pos.reshape(spatial_size,spatial_size,-1).permute(2,0,1)
            img_pos=F.interpolate(img_pos.unsqueeze(0), size=x.shape[-2:], mode='bicubic', align_corners=False)
            img_pos=img_pos.flatten(2,3).permute(2,0,1)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = x + img_pos  # (HW+1)NC
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
        img_feats=ckpt.checkpoint(inner_forward,x) if self.training else inner_forward(x)
        return img_feats
    def text_embeds(self,text):
        # text_feats=ckpt.checkpoint(self.clip_model.encode_text,text)
        text_feats=self.clip_model.encode_text(text)
        text_feats=text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        text_feats=self.bn_neck_text(text_feats)
        return text_feats
    def forward_gallery(self, image_list, gt_instances):
        vfeatures = self.backbone(image_list.tensor)
        b,_,h,w=vfeatures.shape
        vmasks=image_list.mask
        vmasks=F.interpolate(vmasks[None].float(), size=vfeatures.shape[-2:]).to(torch.bool)[0]
        v_pos=self.visual_pos(vfeatures,vmasks)
        vfeatures=self.img_embes(vfeatures)

        if self.training:
            losses={}
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                #TODO fix for dist
                losses["loss_oim_text"]=0.0*vfeatures.sum()
                for i in range(len(self.grounding_trans.resblocks)):
                    losses["loss_conf_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_bbox_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_giou_{}".format(i)]=torch.tensor(0.,device=self.device)
                pred_instances=[]
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss_text(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v
                gt_pids=[inst.gt_pids for inst in gt_instances]
                num_gts=[inst.gt_pids.shape[0] for inst in gt_instances]
                match_indices=[]
                for bi in range(b):
                    bidx_offset=bi*text_pids.shape[0]
                    gidx_offset=0 if bi==0 else sum(num_gts[:bi])
                    bi_match_mask=text_pids.unsqueeze(1)==gt_pids[bi].unsqueeze(0)
                    bi_match_indices=torch.nonzero(bi_match_mask)
                    bi_indices=bi_match_indices+torch.tensor([[bidx_offset,gidx_offset]],device=bi_match_indices.device)
                    match_indices.append(bi_indices)
                match_indices=torch.cat(match_indices)
                # query grounding
                tgt=text_embs.unsqueeze(1).repeat(1, b, 1)
                query_pos=self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
                m_query_out=self.grounding_trans(tgt,vfeatures,vfeatures,vmasks.flatten(1,2),v_pos.flatten(2,3).permute(2,0,1),query_pos,with_ckpt=True,return_inter=True)
                m_out_conf = self.conf_estimate(m_query_out,tgt)
                m_out_bbox = self.bbox_embed(m_query_out).sigmoid()
                grounding_losses=self.m_pred_loss(m_out_conf,m_out_bbox,gt_instances,match_indices)
                losses.update(grounding_losses)
                with torch.no_grad():
                    pred_instances=[]
                    for bi in range(b):
                        inst=Instances(image_list.image_sizes[bi])
                        ccwh_box=m_out_bbox[-1][:,bi].detach()
                        pred_boxes=Boxes(ccwh_box,box_mode=BoxMode.CCWH_REL)
                        inst.pred_boxes=pred_boxes.convert_mode(BoxMode.XYXY_ABS,image_list.image_sizes[bi])
                        inst.pred_scores=F.softmax(m_out_conf[-1][:,bi].detach(),dim=-1)[:,0]
                        inst.pred_classes=text_pids
                        inst.assign_ids=text_pids
                        pred_instances.append(inst)
            return pred_instances, [vfeatures.detach().permute(1,2,0).reshape(b,-1,h,w)], losses
        else:
            raise NotImplementedError

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
class ClipQueryGroundingBaselineDC2(ClipQueryGroundingBaseline):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        res5=self.clip_model.visual.layer4 # without stride
        stride_block=res5[0]
        stride_block.avgpool=nn.Identity()
        stride_block.downsample[0]=nn.Identity()

@META_ARCH_REGISTRY.register()
class ClipQueryGroundingBaselineMetric(ClipQueryGroundingBaseline):
    @classmethod
    def _build_out_embed(cls,embed_dim):
        # NOTE from DETR
        conf_embed =nn.Sequential( nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, 2))
        bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        return conf_embed,bbox_embed
    def conf_estimate(self,m_query_out,query_input):
        m_query_sub_square=torch.pow(m_query_out-query_input.unsqueeze(0),2)
        m,l,b,c=m_query_sub_square.shape
        m_query_sub_square=m_query_sub_square.flatten(0,2)
        return self.conf_embed(m_query_sub_square).reshape(m,l,b,-1)

from psd2.utils.box_augmentation import build_box_augmentor
from psd2.modeling.poolers import ROIPooler
@META_ARCH_REGISTRY.register()
class ClipQueryGroundingBiAlign(ClipQueryGroundingBaseline):
    @configurable
    def __init__(
        self,
        box_augmentor,
        box_pooler,
        oim_loss_img,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args,**kwargs)
        self.box_augmentor = box_augmentor
        self.box_pooler=box_pooler
        self.oim_loss_img=oim_loss_img
        self.alignment_loss=ProtoConLoss()
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        reid_cfg = cfg.PERSON_SEARCH.REID
        reid_pooler_cfg = reid_cfg.ROI_POOLER
        ret["box_augmentor"] = build_box_augmentor(reid_cfg.BOX_AUGMENTATION) if reid_cfg.BOX_AUGMENTATION.ENABLE else None
        ret["box_pooler"] = ROIPooler(
            output_size=reid_pooler_cfg.POOLER_RESOLUTION,
            scales=reid_pooler_cfg.SCALES,
            sampling_ratio=reid_pooler_cfg.SAMP_RATIO,
            pooler_type=reid_pooler_cfg.TYPE,
        )
        ret["oim_loss_img"]=OIMLoss(cfg)
        return ret
    def box_embeds_pids(self,img_features,gt_instances):
        pos_boxes, pos_ids = [], []
        for gts_i in gt_instances:
            # append gt
            pos_boxes.append(gts_i.gt_boxes.tensor)
            pos_ids.append(gts_i.gt_pids)
        if self.box_augmentor is not None:
            pos_boxes, pos_ids = self.box_augmentor.augment_boxes(
                pos_boxes,
                pos_ids,
                det_boxes=None,
                det_pids=None,
                img_sizes=[gti.image_size for gti in gt_instances],
            )# gt_included
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in pos_boxes]
        pos_feats = self.box_pooler([img_features], d2_boxes)
        return pos_feats.mean((2,3)),torch.cat(pos_ids)
        
    def align_loss(self,box_embs,box_pids,text_embs,text_pids):
        losses={}
        lb_features=self.oim_loss_img.lb_layer.lookup_table
        ulb_features=self.oim_loss_img.ulb_layer.queue
        v_features=torch.cat([lb_features,ulb_features],dim=0)
        v_embs=F.normalize(v_features,dim=-1)
        text_embs=F.normalize(text_embs,dim=-1)
        align_loss_t=self.alignment_loss(text_embs,v_embs,text_pids)
        losses["loss_align_t"]=align_loss_t*0.5
        lb_features=self.oim_loss_text.lb_layer.lookup_table
        t_embs=F.normalize(lb_features,dim=-1)
        pos_mask=box_pids>-1
        box_embs=box_embs[pos_mask]
        box_pids=box_pids[pos_mask]
        box_embs=F.normalize(box_embs,dim=-1)
        align_loss_i=self.alignment_loss(box_embs,t_embs,box_pids)
        losses["loss_align_i"]=align_loss_i*0.5
        return losses

    def forward_gallery(self, image_list, gt_instances):
        vfeatures = self.backbone(image_list.tensor)
        b,_,h,w=vfeatures.shape
        vmasks=image_list.mask
        vmasks=F.interpolate(vmasks[None].float(), size=vfeatures.shape[-2:]).to(torch.bool)[0]
        v_pos=self.visual_pos(vfeatures,vmasks)
        vfeatures=self.img_embes(vfeatures)

        if self.training:
            losses={}
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            text_pids=torch.stack(text_pids).to(self.device)
            text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
            reid_loss_text = self.oim_loss_text(text_embs,text_pids )
            for k,v in reid_loss_text.items():
                losses[k+"_text"]=v*0.5
            img_embeds,img_pids=self.box_embeds_pids(vfeatures.permute(1,2,0).reshape(b,-1,h,w),gt_instances)
            reid_loss_img = self.oim_loss_img(img_embeds,img_pids)
            for k,v in reid_loss_img.items():
                losses[k+"_img"]=v*0.5
            if len(text_pids)==0:
                #TODO fix for dist
                for i in range(len(self.grounding_trans.resblocks)):
                    losses["loss_conf_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_bbox_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_giou_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_align_t"]=torch.tensor(0.,device=self.device)
                    losses["loss_align_i"]=torch.tensor(0.,device=self.device)
                pred_instances=[]
            else:
                losses.update(self.align_loss(img_embeds,img_pids,text_embs,text_pids))
                gt_pids=[inst.gt_pids for inst in gt_instances]
                num_gts=[inst.gt_pids.shape[0] for inst in gt_instances]
                match_indices=[]
                for bi in range(b):
                    bidx_offset=bi*text_pids.shape[0]
                    gidx_offset=0 if bi==0 else sum(num_gts[:bi])
                    bi_match_mask=text_pids.unsqueeze(1)==gt_pids[bi].unsqueeze(0)
                    bi_match_indices=torch.nonzero(bi_match_mask)
                    bi_indices=bi_match_indices+torch.tensor([[bidx_offset,gidx_offset]],device=bi_match_indices.device)
                    match_indices.append(bi_indices)
                match_indices=torch.cat(match_indices)
                # query grounding
                tgt=text_embs.unsqueeze(1).repeat(1, b, 1)
                query_pos=self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
                m_query_out=self.grounding_trans(tgt,vfeatures,vfeatures,vmasks.flatten(1,2),v_pos.flatten(2,3).permute(2,0,1),query_pos,with_ckpt=True,return_inter=True)
                m_out_conf = self.conf_estimate(m_query_out,tgt)
                m_out_bbox = self.bbox_embed(m_query_out).sigmoid()
                grounding_losses=self.m_pred_loss(m_out_conf,m_out_bbox,gt_instances,match_indices)
                losses.update(grounding_losses)
                with torch.no_grad():
                    pred_instances=[]
                    for bi in range(b):
                        inst=Instances(image_list.image_sizes[bi])
                        ccwh_box=m_out_bbox[-1][:,bi].detach()
                        pred_boxes=Boxes(ccwh_box,box_mode=BoxMode.CCWH_REL)
                        inst.pred_boxes=pred_boxes.convert_mode(BoxMode.XYXY_ABS,image_list.image_sizes[bi])
                        inst.pred_scores=F.softmax(m_out_conf[-1][bi].detach(),dim=-1)[:,0]
                        inst.pred_classes=text_pids
                        inst.assign_ids=text_pids
                        pred_instances.append(inst)
            return pred_instances, [vfeatures.detach().permute(1,2,0).reshape(b,-1,h,w)], losses
        else:
            return super().forward_gallery(image_list, gt_instances)
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
@META_ARCH_REGISTRY.register()
class ClipQueryGroundingDeform(ClipQueryGroundingBaseline):
    @configurable
    def __init__(
        self,
        ref_points,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.ref_points=ref_points
        self.grounding_trans.bbox_embed=self.bbox_embed
    @classmethod
    def _build_grounding_transformer(cls,embed_dim):
        trans=RefineFullyDeformableCrossAttentionTransformer(width=embed_dim,
                                                    layers=4,
                                                    heads=embed_dim //
                                                    64)
        for p in trans.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in trans.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        return trans
        
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        conf_embed=ret["conf_embed"]
        bbox_embed=ret["bbox_embed"]
        conf_embed=nn.ModuleList([copy.deepcopy(conf_embed) for _ in range(4)])
        bbox_embed=nn.ModuleList([copy.deepcopy(bbox_embed) for _ in range(4)])
        ret["conf_embed"]=conf_embed
        ret["bbox_embed"]=bbox_embed
        embed_dim=ret["clip_model"].text_projection.shape[1]
        ref_pts=nn.Linear(embed_dim, 2)
        xavier_uniform_(ref_pts.weight.data, gain=1.0)
        constant_(ref_pts.bias.data, 0.)
        ret["ref_points"]=ref_pts
        return ret
    @classmethod
    def _build_out_embed(cls,embed_dim):
        # NOTE from Deformable DETR
        conf_embed =nn.Linear(embed_dim, 1)
        bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        return conf_embed,bbox_embed
    def conf_estimate(self,m_query_out,query_input):
        m,l,b,c=m_query_out.shape
        m_conf=[]
        for i,out in enumerate(m_query_out):
            m_conf.append(self.conf_embed[i](out))
        return torch.stack(m_conf)
    def conf_loss(self,scores,gt_instances,match_indices,num_inst):
        bidx,gidx=self._get_src_permutation_idx(match_indices)
        gidx=[0]*len(gidx)
        scores=scores.flatten(0,1)
        targets=torch.zeros_like(scores,dtype=scores.dtype)
        targets[bidx,gidx]=1
        loss=  sigmoid_focal_loss(
                scores,
                targets,
                num_inst,
                alpha=0.25,
                gamma=2,
            )
        return {"loss_conf": loss * 1.0}
    def bbox_loss(self,pred_boxes,gt_instances,match_indices,num_inst):
        # NOTE predictions are cchw in (0,1)
        pred_boxes=pred_boxes.flatten(0,1)
        bidx,gidx=self._get_src_permutation_idx(match_indices)
        src_boxes = pred_boxes[bidx]
        cchw_gt_boxes=torch.cat([inst.gt_boxes.convert_mode(BoxMode.CCWH_REL,inst.image_size).tensor for inst in gt_instances])
        target_boxes = cchw_gt_boxes[gidx]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_inst * 5.0

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_inst *2.0
        return losses
    def forward_gallery(self, image_list, gt_instances):
        vfeatures = self.backbone(image_list.tensor)
        b,_,h,w=vfeatures.shape
        vmasks=image_list.mask
        vmasks=F.interpolate(vmasks[None].float(), size=vfeatures.shape[-2:]).to(torch.bool)[0]
        # v_pos=self.visual_pos(vfeatures,vmasks)
        vfeatures=self.img_embes(vfeatures)

        if self.training:
            losses={}
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            text_pids=torch.stack(text_pids).to(self.device)
            text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
            reid_loss_text = self.oim_loss_text(text_embs,text_pids )
            for k,v in reid_loss_text.items():
                losses[k+"_text"]=v
            if len(text_pids)==0:
                #TODO fix for dist
                for i in range(len(self.grounding_trans.resblocks)):
                    losses["loss_conf_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_bbox_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_giou_{}".format(i)]=torch.tensor(0.,device=self.device)
                pred_instances=[]
            else:
                gt_pids=[inst.gt_pids for inst in gt_instances]
                num_gts=[inst.gt_pids.shape[0] for inst in gt_instances]
                match_indices=[]
                for bi in range(b):
                    bidx_offset=bi*text_pids.shape[0]
                    gidx_offset=0 if bi==0 else sum(num_gts[:bi])
                    bi_match_mask=text_pids.unsqueeze(1)==gt_pids[bi].unsqueeze(0)
                    bi_match_indices=torch.nonzero(bi_match_mask)
                    bi_indices=bi_match_indices+torch.tensor([[bidx_offset,gidx_offset]],device=bi_match_indices.device)
                    match_indices.append(bi_indices)
                match_indices=torch.cat(match_indices)
                # query grounding
                tgt=text_embs.unsqueeze(1).repeat(1, b, 1)
                query_pos=self.query_embed.weight.unsqueeze(1).repeat(tgt.shape[0], b, 1)
                reference_points=self.ref_points(query_pos).sigmoid()
                init_reference_out=reference_points.transpose(0,1).contiguous()
                spatial_shapes=torch.as_tensor([(h,w)], dtype=torch.long, device=vfeatures.device)
                level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
                valid_ratios = torch.stack([get_valid_ratio(vmasks)], 1)
                mask_flatten=vmasks.flatten(1)
                m_query_out, inter_references = self.grounding_trans(tgt.transpose(0,1).contiguous(), reference_points.transpose(0,1).contiguous(), vfeatures.transpose(0,1).contiguous(),
                                            spatial_shapes, level_start_index, valid_ratios, query_pos.transpose(0,1).contiguous(), mask_flatten,with_ckpt=True)
                inter_references_out = inter_references
                m_out_conf = self.conf_estimate(m_query_out,tgt)
                m_out_bbox = []
                for lvl in range(m_query_out.shape[0]):
                    if lvl == 0:
                        reference = init_reference_out
                    else:
                        reference = inter_references[lvl - 1]
                    reference = inverse_sigmoid(reference)
                    tmp = self.bbox_embed[lvl](m_query_out[lvl])
                    if reference.shape[-1] == 4:
                        tmp += reference
                    else:
                        assert reference.shape[-1] == 2
                        tmp[..., :2] += reference
                    outputs_coord = tmp.sigmoid()
                    m_out_bbox.append(outputs_coord)
                m_out_bbox= torch.stack(m_out_bbox)
                grounding_losses=self.m_pred_loss(m_out_conf,m_out_bbox,gt_instances,match_indices)
                losses.update(grounding_losses)
                with torch.no_grad():
                    pred_instances=[]
                    for bi in range(b):
                        inst=Instances(image_list.image_sizes[bi])
                        ccwh_box=m_out_bbox[-1][bi].detach()
                        pred_boxes=Boxes(ccwh_box,box_mode=BoxMode.CCWH_REL)
                        inst.pred_boxes=pred_boxes.convert_mode(BoxMode.XYXY_ABS,image_list.image_sizes[bi])
                        inst.pred_scores=m_out_conf[-1][bi].detach().sigmoid()[:,0]
                        inst.pred_classes=text_pids
                        inst.assign_ids=text_pids
                        pred_instances.append(inst)
            return pred_instances, [vfeatures.detach().permute(1,2,0).reshape(b,-1,h,w)], losses
        else:
            raise NotImplementedError




class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

#TODO test post norm
class ResidualFullyCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,v_key_padding_mask=None,pos=None,q_pos=None):
        # cross attn
        q = q + self.attn(self.with_pos_embed(self.ln_1(q),q_pos),self.with_pos_embed(k,pos),v,need_weights=False, attn_mask=None,key_padding_mask=v_key_padding_mask)[0]
        q = q + self.mlp(self.ln_2(q))
        return q

class FullyCrossAttentionTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualFullyCrossAttentionBlock(width, heads) for _ in range(layers)])
    def forward(self, q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,memory_key_padding_mask= None,
                pos = None,
                query_pos= None,with_ckpt=False,return_inter=False):
        inter_out=[]

        if with_ckpt:
            # def inner_forward()
            for blk in self.resblocks:
                q=ckpt.checkpoint(blk,q,k,v,memory_key_padding_mask,pos,query_pos)
                if return_inter:
                    inter_out.append(q)
        else:
            for blk in self.resblocks:
                q=blk(q,k,v,memory_key_padding_mask,pos,query_pos)
                if return_inter:
                    inter_out.append(q)
        if return_inter:
            q=torch.stack(inter_out)
        return q

#NOTE post norm by default
from psd2.layers.ms_deform_attn import MSDeformAttn
class ResidualFullyDeformableCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_levels: int,n_points: int):
        super().__init__()
        self.attn = MSDeformAttn(d_model, n_levels, n_head, n_points)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.ReLU(inplace=True)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # cross attn
        tgt2 = self.attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + tgt2
        tgt = self.ln_1(tgt)
        # ffn
        tgt = tgt+ self.mlp(tgt)
        tgt = self.ln_2(tgt)
        return tgt

class RefineFullyDeformableCrossAttentionTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualFullyDeformableCrossAttentionBlock(width, heads,1,4,) for _ in range(layers)])
        self.box_embed=None # assign by parent module
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None,with_ckpt=False):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.resblocks):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if with_ckpt:
                output =ckpt.checkpoint(layer,output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            else:
                output =layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            # hack implementation for iterative bounding box refinement
            tmp = self.bbox_embed[lid](output)
            if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
            else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)



import math
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

    def forward(self, x,mask):
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
from torchvision.ops.boxes import box_area
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

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

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio