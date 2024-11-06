import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import logging
from .base_tbps import SearchBaseTBPS
from ..build import META_ARCH_REGISTRY
from psd2.config.config import configurable
from psd2.structures import Boxes, Instances, BoxMode
from psd2.layers.mem_matching_losses import OIMLoss
from psd2.modeling.extend.clip_model import build_CLIP_from_openai_pretrained
import copy
from collections import OrderedDict
import torch.utils.checkpoint as ckpt
import psd2.utils.comm as comm
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# GeneralizedRCNN as reference
@META_ARCH_REGISTRY.register()
class MQueryGroundingBaseline(SearchBaseTBPS):
    @configurable
    def __init__(
        self,
        clip_model,
        visual_pos,
        grounding_transformer,
        num_query,
        rand_modal,
        bbox_embed,
        conf_embed,
        bn_neck,
        oim_loss_fuse,
        qimg_size,
        n_query_infer,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.clip_model=clip_model
        self.oim_loss_fuse=oim_loss_fuse
        self.bn_neck = bn_neck
        self.bn_neck_text=copy.deepcopy(bn_neck)
        self.grounding_trans=grounding_transformer
        self.bbox_embed=bbox_embed
        self.conf_embed=conf_embed
        self.query_embed = nn.Embedding(num_query, clip_model.text_projection.shape[1])
        self.visual_pos=visual_pos

        # query encoder
        self.rand_modal=rand_modal
        self.qimg_tokenizer=nn.Conv2d(in_channels=3, out_channels=clip_model.token_embedding.weight.shape[1], kernel_size=16, stride=16, bias=False)
        qh,qw=qimg_size
        scale = clip_model.token_embedding.weight.shape[1] ** -0.5 
        self.qimg_pos_embed=nn.Parameter(scale * torch.randn( qh//16*qw//16, clip_model.token_embedding.weight.shape[1]))
        self.query_cls_token = nn.Parameter(scale * torch.randn(1, clip_model.token_embedding.weight.shape[1]))
        self.query_cls_pos = nn.Parameter(scale * torch.randn(1, clip_model.token_embedding.weight.shape[1]))

        # grounding infer
        self.n_query_infer=n_query_infer
        # maych /loss weights
        self.cls_weight=1.0
        self.bbox_weight=5.0
        self.giou_weight=2.0
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        clip_model,_=build_CLIP_from_openai_pretrained(cfg.PERSON_SEARCH.DET.CLIP.NAME,cfg.PERSON_SEARCH.DET.CLIP.IMG_SIZE,cfg.PERSON_SEARCH.DET.CLIP.STRIDE,text_dropout=cfg.PERSON_SEARCH.DET.CLIP.TEXT_DROPOUT)
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
        oim_loss=OIMLoss(cfg)
        ret["oim_loss_fuse"]=oim_loss
        embed_dim=clip_model.text_projection.shape[1]
        ret["visual_pos"]=PositionEmbeddingSine(embed_dim//2,normalize=True)
        ret["grounding_transformer"]=cls._build_grounding_transformer(embed_dim)
        conf_embed,bbox_embed=cls._build_out_embed(embed_dim)
        ret["conf_embed"]=conf_embed
        ret["bbox_embed"]=bbox_embed
        ret["qimg_size"]=cfg.INPUT.QUERY_SIZE
        ret["n_query_infer"]=cfg.PERSON_SEARCH.REID.MODEL.NUM_INFER_QUERY
        ret["num_query"]=cfg.PERSON_SEARCH.REID.MODEL.NUM_QUERY
        ret["rand_modal"]=cfg.PERSON_SEARCH.REID.MODEL.RAND_MODAL
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
    
    def m_pred_loss(self,m_query_conf,m_query_bbox,m_query_pids,gt_instances):
        match_indices=self.m_set_match(m_query_conf,m_query_bbox,m_query_pids,gt_instances)
        losses={}
        num_pos=sum([mbi[0].shape[0] for mbi in match_indices[-1]])
        
        if comm.get_world_size()>1:
            all_num_pos=comm.all_gather(num_pos)
            num_pos = sum( all_num_pos)
            num_pos=max(num_pos / comm.get_world_size(),1)

        for i in range(len(self.grounding_trans.resblocks)):
            ilosses=self.conf_loss(m_query_conf[i],gt_instances,match_indices[i],num_pos)
            ilosses.update(self.bbox_loss(m_query_bbox[i],gt_instances,match_indices[i],num_pos))
            for k,v in ilosses.items():
                losses[k+"_{}".format(i)]=v
        return losses
    def m_set_match(self,m_query_conf,m_query_bbox,m_query_pids,gt_instances):
        raise NotImplementedError
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

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
    def query_embeds(self,text,img):
        # text_feats=ckpt.checkpoint(self.clip_model.encode_text,text)
        if img is None:
            q_feat=self.clip_model.token_embedding(text)+self.clip_model.positional_embedding
        elif text is None:
            q_feat=self.qimg_tokenizer(img).flatten(2,3).transpose(1,2)+self.qimg_pos_embed.unsqueeze(0)
        else:
            q_feat_t=self.clip_model.token_embedding(text)+self.clip_model.positional_embedding
            q_feat_i=self.qimg_tokenizer(img).flatten(2,3).transpose(1,2)+self.qimg_pos_embed.unsqueeze(0)
            q_feat=torch.cat([q_feat_i,q_feat_t],dim=1)
        q_cls=(self.query_cls_token+self.query_cls_pos).unsqueeze(0).expand(q_feat.shape[0],-1,-1)
        q_feat=torch.cat([q_cls,q_feat],dim=1)
        q_feat = q_feat.permute(1, 0, 2)  # NLD -> LND
        q_feat_attn_mask=torch.zeros((q_feat.shape[0],q_feat.shape[0]),dtype=q_feat.dtype,device=self.device)
        if text is not None:
            q_feat_attn_mask[-text.shape[1]:,-text.shape[1]:]=torch.empty(self.clip_model.context_length, self.clip_model.context_length).fill_(float("-inf")).triu_(1).to(self.device)
            text_idx_offset=1+(0 if img is None else self.qimg_pos_embed.shape[0])
            text_cls_idxs=text.argmax(dim=-1)
            n_heads=self.clip_model.ln_final.weight.shape[0]//64
            n_query=q_feat.shape[1]
            q_feat_attn_mask=q_feat_attn_mask.repeat(n_query,1,1)
            for bi,idx in enumerate(text_cls_idxs): # mask padded text tokens
                q_feat_attn_mask[bi,:,text_idx_offset+idx+1:]=float("-inf")
            q_feat_attn_mask=q_feat_attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
        q_feat = self.clip_model.transformer(q_feat,True,q_feat_attn_mask)
        q_feat = q_feat.permute(1, 0, 2)  # LND -> NLD
        q_feat = self.clip_model.ln_final(q_feat).type(self.clip_model.dtype)
        q_feat = q_feat @ self.clip_model.text_projection
        q_feat = q_feat[:,0]
        q_feat = self.bn_neck_query(q_feat)
        return q_feat
    def bn_neck_query(self,x):
        return self.bn_neck_text(x)
    
    def forward(self, input_list, query_embeds=None):
        """
        preds:
            a list of
            {
                "pred_boxes": XYXY_ABS Boxes in augmentation range (resizing/cropping/flipping/...) during training
                            XYXY_ABS Boxes in original range for test
                "pred_scores": tensor
                "assign_ids": assigned person identities (during training only)
                "reid_feats": tensor
            }
        """
        if "query" in input_list[0].keys():
            q_img_list = self.preprocess_input([qi["query"] for qi in input_list])
            q_gt_instances = [
                gti["query"]["instances"].to(self.device) for gti in input_list
            ]
            q_outs = self.forward_query(q_img_list, q_gt_instances)
            return q_outs
        else:
            image_list = self.preprocess_input(input_list)
            gt_instances = [gti["instances"].to(self.device) for gti in input_list]
            if self.training:
                preds, feat_maps, losses = self.forward_gallery(
                    image_list, gt_instances
                )
                self.visualize_training(image_list, gt_instances, preds, feat_maps)
                return losses
            else:
                return self.infer_grounding(image_list, query_embeds.to(self.device))

    def infer_grounding(self,image_list,query_embeds):
        vfeatures = self.backbone(image_list.tensor)
        b,_,h,w=vfeatures.shape
        vmasks=image_list.mask
        vmasks=F.interpolate(vmasks[None].float(), size=vfeatures.shape[-2:]).to(torch.bool)[0]
        v_pos=self.visual_pos(vfeatures,vmasks)
        vfeatures=self.img_embes(vfeatures)
        infer_results=[]
        raise NotImplementedError

        

    def forward_gallery(self, image_list, gt_instances):
        raise NotImplementedError
        vfeatures = self.backbone(image_list.tensor)
        b,_,h,w=vfeatures.shape
        vmasks=image_list.mask
        vmasks=F.interpolate(vmasks[None].float(), size=vfeatures.shape[-2:]).to(torch.bool)[0]
        v_pos=self.visual_pos(vfeatures,vmasks)
        vfeatures=self.img_embes(vfeatures)

        if self.training:
            losses={}
            # text oim
            text_tokens,text_pids,query_imgs,qimg_pids,qimg_paired_text_tokens=[],[],[],[],[]
            for inst in gt_instances:
                for texts,pid,qimg in zip(inst.descriptions,inst.gt_pids,inst.queries):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
                        query_imgs.append(qimg)
                        qimg_pids.append(pid)
                        qimg_paired_text_tokens.append(texts[torch.randint(0,num_text,(1,))[0]])
            if len(text_pids)==0:
                #TODO fix for dist
                losses["loss_oim_text"]=0.0*vfeatures.sum()
                losses["loss_oim_qimg"]=0.0*vfeatures.sum()
                losses["loss_oim_fuse"]=0.0*vfeatures.sum()
                for i in range(len(self.grounding_trans.resblocks)):
                    losses["loss_conf_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_bbox_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_giou_{}".format(i)]=torch.tensor(0.,device=self.device)
                pred_instances=[]
            else:
                # text query
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.query_embeds(torch.stack(text_tokens).to(self.device),None)
                reid_loss_text = self.oim_loss_fuse(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v
                # image query
                qimg_pids=torch.stack(qimg_pids).to(self.device)
                qimg_embs=self.query_embeds(None,torch.stack(query_imgs).to(self.device))
                reid_loss_qimg = self.oim_loss_fuse(qimg_embs,qimg_pids )
                for k,v in reid_loss_qimg.items():
                    losses[k+"_qimg"]=v
                # fused query
                fuse_pids=torch.stack(qimg_pids).to(self.device)
                fuse_embs=self.query_embeds(torch.stack(qimg_paired_text_tokens).to(self.device),torch.stack(query_imgs).to(self.device))
                reid_loss_fuse = self.oim_loss_fuse(fuse_embs,fuse_pids )
                for k,v in reid_loss_fuse.items():
                    losses[k+"_fuse"]=v
                query_embs=torch.cat([text_embs,qimg_embs,fuse_embs],dim=0)
                query_pids=torch.cat([text_pids,qimg_pids,fuse_pids],dim=0)
                """
                gt_pids=[inst.gt_pids for inst in gt_instances]
                num_gts=[inst.gt_pids.shape[0] for inst in gt_instances]
                
                match_indices=[]
                for bi in range(b):
                    bidx_offset=bi*query_pids.shape[0]
                    gidx_offset=0 if bi==0 else sum(num_gts[:bi])
                    bi_match_mask=query_pids.unsqueeze(1)==gt_pids[bi].unsqueeze(0)
                    bi_match_indices=torch.nonzero(bi_match_mask)
                    bi_indices=bi_match_indices+torch.tensor([[bidx_offset,gidx_offset]],device=bi_match_indices.device)
                    match_indices.append(bi_indices)
                match_indices=torch.cat(match_indices)"""
                # query grounding
                n_query_pos=self.query_embed.weight.shape[0]
                n_query=query_embs.shape[0]
                tgt=query_embs.unsqueeze(1).unsqueeze(1).repeat(1,n_query_pos, b, 1).flatten(0,1)
                query_pos=self.query_embed.weight.unsqueeze(1).unsqueeze(0).repeat(n_query,1,b, 1).flatten(0,1)
                m_query_out=self.grounding_trans(tgt,vfeatures,vfeatures,vmasks.flatten(1,2),v_pos.flatten(2,3).permute(2,0,1),query_pos,with_ckpt=True,return_inter=True)
                m_out_conf = self.conf_estimate(m_query_out,tgt)
                m_out_bbox = self.bbox_embed(m_query_out).sigmoid()
                grounding_losses=self.m_pred_loss(m_out_conf,m_out_bbox,gt_instances)
                losses.update(grounding_losses)
                pred_pids_per_img=query_pids.unsqueeze(1).repeat(1,n_query_pos).flatten(0,1)
                with torch.no_grad():
                    pred_instances=[]
                    for bi in range(b):
                        inst=Instances(image_list.image_sizes[bi])
                        ccwh_box=m_out_bbox[-1][:,bi].detach()
                        pred_boxes=Boxes(ccwh_box,box_mode=BoxMode.CCWH_REL)
                        inst.pred_boxes=pred_boxes.convert_mode(BoxMode.XYXY_ABS,image_list.image_sizes[bi])
                        inst.pred_scores=F.softmax(m_out_conf[-1][:,bi].detach(),dim=-1)[:,0]
                        inst.pred_classes=pred_pids_per_img
                        inst.assign_ids=pred_pids_per_img
                        pred_instances.append(inst)
            return pred_instances, [vfeatures.detach().permute(1,2,0).reshape(b,-1,h,w)], losses
        else:
            raise NotImplementedError

    def forward_query(self, image_list, gt_instances):
        # one sentece for each query
        text_tokens=[]
        qimgs=[]
        for inst in gt_instances:
            if hasattr(inst,"descriptions"):
                text_tokens.append(inst.descriptions[0][0][0])
            if hasattr(inst,"queries"):
                qimgs.append(inst.queries[0][0])
        if len(text_tokens)==0:
            text_tokens=None
        else:
            text_tokens=torch.stack(text_tokens).to(self.device)
        if len(qimgs)==0:
            qimgs=None
        else:
            qimgs=torch.stack(qimgs).to(self.device)
        query_embs=self.query_embeds(text_tokens,qimgs)
        query_embs = torch.split(query_embs, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=query_embs[i])
            for i in range(len(query_embs))
        ]


@META_ARCH_REGISTRY.register()
class MQueryGroundingMetric(MQueryGroundingBaseline):
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
class MQueryGroundingBiAlign(MQueryGroundingBaseline): #TODO refactor
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
        text_embs=F.normalize(text_embs,dim=-1)
        pos_mask=box_pids>-1
        box_embs=box_embs[pos_mask]
        box_pids=box_pids[pos_mask]
        box_embs=F.normalize(box_embs,dim=-1)
        losses["loss_align"]=self.compute_sdm(box_embs,text_embs,box_pids,text_pids,1/0.02)
        return losses

    def compute_sdm(self,features1, features2, pid1,pid2, logit_scale, epsilon=1e-8):
        pid1_set=set(pid1.cpu().numpy().tolist())
        pid2_set=set(pid2.cpu().numpy().tolist())
        remove_id_1=pid1_set-pid2_set
        remove_id_2=pid2_set-pid1_set
        if len(remove_id_1)>0:
            keep_mask1=torch.ones(features1.shape[0])
            for rid in list(remove_id_1):
                keep_mask1[pid1==rid]=0.
            keep_mask1=keep_mask1==1
            features1=features1[keep_mask1]
            pid1=pid1[keep_mask1]
        if len(remove_id_2)>0:
            keep_mask2=torch.ones(features2.shape[0])
            for rid in list(remove_id_2):
                keep_mask2[pid2==rid]=0.
            keep_mask2=keep_mask2==1
            features2=features2[keep_mask2]
            pid2=pid2[keep_mask2]
        """
        Similarity Distribution Matching
        """
        pid1 = pid1.reshape((features1.shape[0], 1)) # make sure pid size is [batch_size, 1]
        pid2=pid2.reshape((features2.shape[0], 1)) # make sure pid size is [batch_size, 1]
        pid_dist = pid1 - pid2.t() # n1 x n2
        labels_1_2 = (pid_dist == 0).float()
        labels_2_1 = labels_1_2.t()

        f1_norm = features1 / features1.norm(dim=1, keepdim=True)
        f2_norm = features2 / features2.norm(dim=1, keepdim=True)

        f2f1_cosine_theta = f2_norm @ f1_norm.t()
        f1f2_cosine_theta = f2f1_cosine_theta.t()

        f2_proj_f1 = logit_scale * f2f1_cosine_theta
        f1_proj_f2 = logit_scale * f1f2_cosine_theta

        # normalize the true matching distribution
        labels_1_2_distribute = labels_1_2 / labels_1_2.sum(dim=1,keepdim=True)
        labels_2_1_distribute = labels_2_1 / labels_2_1.sum(dim=1,keepdim=True)

        f1f2_pred = F.softmax(f1_proj_f2, dim=1)
        f1f2_loss = f1f2_pred * (F.log_softmax(f1_proj_f2, dim=1) - torch.log(labels_1_2_distribute + epsilon))
        f2f1_pred = F.softmax(f2_proj_f1, dim=1)
        f2f1_loss = f2f1_pred * (F.log_softmax(f2_proj_f1, dim=1) - torch.log(labels_2_1_distribute + epsilon))
        # if torch.isnan(f1f2_loss).sum()>0 or torch.isnan(f2f1_loss).sum()>0:
        #    print("get")
        loss = torch.mean(torch.sum(f1f2_loss, dim=1)) + torch.mean(torch.sum(f2f1_loss, dim=1))

        return loss

    def forward_gallery(self, image_list, gt_instances): #TODO
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
            reid_loss_text = self.oim_loss(text_embs,text_pids )
            for k,v in reid_loss_text.items():
                losses[k+"_text"]=v*0.5
            img_embeds,img_pids=self.box_embeds_pids(vfeatures.permute(1,2,0).reshape(b,-1,h,w),gt_instances)
            reid_loss_img = self.oim_loss(img_embeds,img_pids)
            for k,v in reid_loss_img.items():
                losses[k+"_img"]=v*0.5
            if len(text_pids)==0:
                #TODO fix for dist
                for i in range(len(self.grounding_trans.resblocks)):
                    losses["loss_conf_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_bbox_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_giou_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_align"]=torch.tensor(0.,device=self.device)
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
class MQueryGroundingDeform(MQueryGroundingBaseline):
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
        bidx,pred_idx=self._get_src_permutation_idx(match_indices)
        targets=torch.zeros_like(scores,dtype=scores.dtype)
        targets[bidx,pred_idx]=1
        loss=  sigmoid_focal_loss(
                scores.flatten(0,1),
                targets.flatten(0,1),
                num_inst,
                alpha=0.25,
                gamma=2,
            )
        return {"loss_conf": loss * 1.0}
    def bbox_loss(self,pred_boxes,gt_instances,match_indices,num_inst):
        # NOTE predictions are cchw in (0,1)
        bidx,pred_idx=self._get_src_permutation_idx(match_indices)
        src_boxes = pred_boxes[bidx,pred_idx]
        gt_ccwh_ref=[inst.gt_boxes.convert_mode(BoxMode.CCWH_REL,inst.image_size).tensor[inst.gt_pids>-1] for inst in gt_instances]
        bidx,match_idx=self._get_tgt_permutation_idx(match_indices)
        target_boxes = torch.stack([gt_ccwh_ref[bi][mi] for bi,mi in zip(bidx,match_idx)])
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_inst * 5.0

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_inst *2.0
        return losses
    @torch.no_grad()
    def m_set_match(self,m_query_conf,m_query_bbox,m_query_pids,gt_instances):
        m_match_indices=[]
        nq=m_query_pids.shape[2]
        n_pos=self.query_embed.weight.shape[0]
        bs=len(gt_instances)
        n_gts = [(inst.gt_pids>-1).shape[0] for inst in gt_instances]
        gt_pids=torch.cat([inst.gt_pids[inst.gt_pids>-1] for inst in gt_instances])
        gt_ccwh_ref=torch.cat([inst.gt_boxes.convert_mode(BoxMode.CCWH_REL,inst.image_size).tensor[inst.gt_pids>-1] for inst in gt_instances])
        gt_xyxy=torch.cat([ inst.gt_boxes.tensor[inst.gt_pids>-1] for inst in gt_instances])
        for query_conf,query_bbox,query_pids in zip(m_query_conf,m_query_bbox,m_query_pids):
            # mask
            pid_not_match=query_pids.flatten(0,1).unsqueeze(1)!=gt_pids.unsqueeze(0)
            # cls_cost
            out_prob=query_conf.sigmoid().flatten(0,1)
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = (pos_cost_class - neg_cost_class).repeat(1,sum(n_gts))
            
            
            # bbox cost
            cost_bbox=torch.cdist(query_bbox.flatten(0,1), gt_ccwh_ref, p=1)
           
            # giou cost
            query_bbox_Boxes=[Boxes(boxes,box_mode=BoxMode.CCWH_REL) for boxes in query_bbox]
            pred_xyxy=torch.cat([boxes.convert_mode(BoxMode.XYXY_ABS,inst.image_size).tensor for boxes,inst in zip(query_bbox_Boxes,gt_instances)])
            cost_giou = -generalized_box_iou(pred_xyxy, gt_xyxy)
            

            cost_all=self.bbox_weight * cost_bbox + self.cls_weight * cost_class + self.giou_weight * cost_giou

            cost_all[pid_not_match]=1e12

            cost_all_bs_nuq=cost_all.view(bs,nq//n_pos,n_pos,-1)
            cost_all_bs_nuq = [c[i] for i, c in enumerate(cost_all_bs_nuq.split(n_gts, -1))]

            indices=[]
            gt_pids_bs=[ pids.cpu().numpy().tolist() for pids in gt_pids.split(n_gts,0)]
            src_idx=torch.arange(0,nq,dtype=torch.long).reshape(nq//n_pos,n_pos).numpy().tolist()
            for cost_bi,pids_bi_g,pids_bi_q in zip(cost_all_bs_nuq,gt_pids_bs,query_pids): # nq//n_pos. n_pos, gi
                pids_bi_q=pids_bi_q.view(-1,n_pos)
                bi_match_srcs=[]
                bi_match_tgts=[]
                for qi in range(nq//n_pos):
                    qpid=pids_bi_q[qi][0].item()
                    if qpid in pids_bi_g:
                        tgt_i=pids_bi_g.index(qpid)
                        src_i=torch.argmin(cost_bi[qi][:,tgt_i]).item()
                        bi_match_srcs.append(src_idx[qi][src_i])
                        bi_match_tgts.append(tgt_i)
                indices.append((bi_match_srcs,bi_match_tgts))

            match_indices= [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]
            m_match_indices.append(match_indices)
        return m_match_indices

    def infer_grounding(self,image_list,query_embeds):
        vfeatures = self.backbone(image_list.tensor)
        b,_,h,w=vfeatures.shape
        vmasks=image_list.mask
        vmasks=F.interpolate(vmasks[None].float(), size=vfeatures.shape[-2:]).to(torch.bool)[0]
        # v_pos=self.visual_pos(vfeatures,vmasks)
        vfeatures=self.img_embes(vfeatures)
        infer_boxes=[[] for _ in range(b)]
        infer_scores=[[] for _ in range(b)]
        # query grounding
        for query_embs in torch.split(query_embeds,self.n_query_infer,dim=1):
            n_query_pos=self.query_embed.weight.shape[0]
            n_query=query_embs.shape[1]
            tgt=query_embs.unsqueeze(2).repeat(1,1,n_query_pos, 1).flatten(1,2)
            query_pos=self.query_embed.weight.unsqueeze(1).unsqueeze(0).repeat(n_query,1,b, 1).flatten(0,1)
            reference_points=self.ref_points(query_pos).sigmoid().transpose(0,1).contiguous()
            init_reference_out=reference_points
            spatial_shapes=torch.as_tensor([(h,w)], dtype=torch.long, device=vfeatures.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([get_valid_ratio(vmasks)], 1)
            mask_flatten=vmasks.flatten(1)
            m_query_out, inter_references = self.grounding_trans(tgt, reference_points, vfeatures.transpose(0,1).contiguous(),
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
            for bi in range(b):
                out_conf=m_out_conf[-1][bi].detach().sigmoid()[...,0].reshape(n_query,n_query_pos)
                out_bbox=m_out_bbox[-1][bi].detach().reshape(n_query,n_query_pos,-1)
                max_conf_per_id=torch.argmax(out_conf,dim=1)
                id_indices=torch.arange(n_query)

                ccwh_box=out_bbox[id_indices,max_conf_per_id]
                pred_boxes=Boxes(ccwh_box,box_mode=BoxMode.CCWH_REL)
                pred_boxes=pred_boxes.convert_mode(BoxMode.XYXY_ABS,image_list.image_sizes[bi])
                pred_scores=out_conf[id_indices,max_conf_per_id]
                infer_boxes[bi].append(pred_boxes.tensor)
                infer_scores[bi].append(pred_scores)
        infer_boxes=[torch.cat(infer_boxes[i],dim=0) for i in range(b)]
        infer_scores=[torch.cat(infer_scores[i],dim=0) for i in range(b)]
        pred_instances=[]
        for bi in range(b):
            inst=Instances(image_list.image_sizes[bi])
            inst.pred_boxes=Boxes(infer_boxes[bi],box_mode=BoxMode.XYXY_ABS)
            inst.pred_scores=infer_scores[bi]
            pred_instances.append(inst)
        return pred_instances
    def query_train(self,text_pids,qimg_pids,text_tokens,query_imgs,qimg_paired_text_tokens):
        losses={}
        if self.rand_modal:
                    select=torch.randint(0,3,(1,)).item()
                    if select==0:
                        # text query
                        text_embs=self.query_embeds(text_tokens,None)
                        reid_loss_text = self.oim_loss_fuse(text_embs,text_pids )
                        for k,v in reid_loss_text.items():
                            losses[k+"_text"]=v
                        query_embs=text_embs
                        query_pids=text_pids
                    elif select==1:
                        # image query
                        qimg_embs=self.query_embeds(None,query_imgs)
                        reid_loss_qimg = self.oim_loss_fuse(qimg_embs,qimg_pids )
                        for k,v in reid_loss_qimg.items():
                            losses[k+"_qimg"]=v
                        query_embs=qimg_embs
                        query_pids=qimg_pids
                    else:
                        # fused query
                        fuse_pids=qimg_pids
                        fuse_embs=self.query_embeds(qimg_paired_text_tokens,query_imgs)
                        reid_loss_fuse = self.oim_loss_fuse(fuse_embs,fuse_pids )
                        for k,v in reid_loss_fuse.items():
                            losses[k+"_fuse"]=v
                        query_embs=fuse_embs
                        query_pids=fuse_pids
        else:
                    # text query
                    text_embs=self.query_embeds(text_tokens,None)
                    reid_loss_text = self.oim_loss_fuse(text_embs,text_pids )
                    for k,v in reid_loss_text.items():
                        losses[k+"_text"]=v
                    # image query
                    qimg_embs=self.query_embeds(None,query_imgs)
                    reid_loss_qimg = self.oim_loss_fuse(qimg_embs,qimg_pids )
                    for k,v in reid_loss_qimg.items():
                        losses[k+"_qimg"]=v
                    # fused query
                    fuse_pids=qimg_pids
                    fuse_embs=self.query_embeds(qimg_paired_text_tokens,query_imgs)
                    reid_loss_fuse = self.oim_loss_fuse(fuse_embs,fuse_pids )
                    for k,v in reid_loss_fuse.items():
                        losses[k+"_fuse"]=v
                    query_embs=torch.cat([text_embs,qimg_embs,fuse_embs],dim=0)
                    query_pids=torch.cat([text_pids,qimg_pids,fuse_pids],dim=0)
        return losses,query_embs,query_pids
    def forward_gallery(self, image_list, gt_instances): # TODO update
        vfeatures = self.backbone(image_list.tensor)
        b,_,h,w=vfeatures.shape
        vmasks=image_list.mask
        vmasks=F.interpolate(vmasks[None].float(), size=vfeatures.shape[-2:]).to(torch.bool)[0]
        # v_pos=self.visual_pos(vfeatures,vmasks)
        vfeatures=self.img_embes(vfeatures)

        if self.training:
            losses={}
            # text oim
            text_tokens,text_pids,query_imgs,qimg_pids,qimg_paired_text_tokens=[],[],[],[],[]
            for inst in gt_instances:
                for texts,pid,qimgs in zip(inst.descriptions,inst.gt_pids,inst.queries):
                    if pid >-1:
                        for qtexts,qimg in zip(texts,qimgs):
                            num_text=len(qtexts)
                            text_tokens.append(qtexts[torch.randint(0,num_text,(1,))[0]])
                            text_pids.append(pid)
                            query_imgs.append((qimg.to(self.device) - self.pixel_mean) / self.pixel_std)
                            qimg_pids.append(pid)
                            qimg_paired_text_tokens.append(qtexts[torch.randint(0,num_text,(1,))[0]])
            if len(text_pids)==0:
                #TODO fix for dist
                losses["loss_oim_text"]=0.0*vfeatures.sum()
                losses["loss_oim_qimg"]=0.0*vfeatures.sum()
                losses["loss_oim_fuse"]=0.0*vfeatures.sum()
                for i in range(len(self.grounding_trans.resblocks)):
                    losses["loss_conf_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_bbox_{}".format(i)]=torch.tensor(0.,device=self.device)
                    losses["loss_giou_{}".format(i)]=torch.tensor(0.,device=self.device)
                pred_instances=[]
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                qimg_pids=torch.stack(qimg_pids).to(self.device)
                
                losses,query_embs,query_pids=self.query_train(text_pids,qimg_pids,torch.stack(text_tokens).to(self.device),torch.stack(query_imgs).to(self.device),torch.stack(qimg_paired_text_tokens).to(self.device))
                
                # query grounding
                n_query_pos=self.query_embed.weight.shape[0]
                n_query=query_embs.shape[0]
                tgt=query_embs.unsqueeze(1).unsqueeze(1).repeat(1,n_query_pos, b, 1).flatten(0,1)
                query_pos=self.query_embed.weight.unsqueeze(1).unsqueeze(0).repeat(n_query,1,b, 1).flatten(0,1)
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
                tgt_pids=query_pids.unsqueeze(1).unsqueeze(1).expand(-1,n_query_pos, b).flatten(0,1)
                m_query_pids=tgt_pids.transpose(0,1).contiguous().unsqueeze(0).expand(m_query_out.shape[0],-1,-1)
                grounding_losses=self.m_pred_loss(m_out_conf,m_out_bbox,m_query_pids,gt_instances)
                losses.update(grounding_losses)
                with torch.no_grad():
                    pred_instances=[]
                    for bi in range(b):
                        inst=Instances(image_list.image_sizes[bi])
                        out_conf=m_out_conf[-1][bi].detach().sigmoid()[...,0].reshape(n_query,n_query_pos)
                        out_bbox=m_out_bbox[-1][bi].detach().reshape(n_query,n_query_pos,-1)
                        max_conf_per_id=torch.argmax(out_conf,dim=1)
                        id_indices=torch.arange(n_query)

                        ccwh_box=out_bbox[id_indices,max_conf_per_id]
                        pred_boxes=Boxes(ccwh_box,box_mode=BoxMode.CCWH_REL)
                        inst.pred_boxes=pred_boxes.convert_mode(BoxMode.XYXY_ABS,image_list.image_sizes[bi])
                        inst.pred_scores=out_conf[id_indices,max_conf_per_id]
                        inst.pred_classes=query_pids
                        inst.assign_ids=query_pids
                        pred_instances.append(inst)
            return pred_instances, [vfeatures.detach().permute(1,2,0).reshape(b,-1,h,w)], losses
        else:
            raise NotImplementedError

@META_ARCH_REGISTRY.register()
class MQueryGrounding8Deform(MQueryGroundingDeform):
    @classmethod
    def _build_grounding_transformer(cls,embed_dim):
        trans=RefineFullyDeformableCrossAttentionTransformer(width=embed_dim,
                                                    layers=8,
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
        ret = super(MQueryGroundingDeform,cls).from_config(cfg)
        conf_embed=ret["conf_embed"]
        bbox_embed=ret["bbox_embed"]
        conf_embed=nn.ModuleList([copy.deepcopy(conf_embed) for _ in range(8)])
        bbox_embed=nn.ModuleList([copy.deepcopy(bbox_embed) for _ in range(8)])
        ret["conf_embed"]=conf_embed
        ret["bbox_embed"]=bbox_embed
        embed_dim=ret["clip_model"].text_projection.shape[1]
        ref_pts=nn.Linear(embed_dim, 2)
        xavier_uniform_(ref_pts.weight.data, gain=1.0)
        constant_(ref_pts.bias.data, 0.)
        ret["ref_points"]=ref_pts
        return ret


@META_ARCH_REGISTRY.register()
class MQueryGrounding8DeformText(MQueryGrounding8Deform):
    def query_train(self,text_pids,qimg_pids,text_tokens,query_imgs,qimg_paired_text_tokens):
        losses={}
        text_embs=self.query_embeds(text_tokens,None)
        reid_loss_text = self.oim_loss_fuse(text_embs,text_pids )
        for k,v in reid_loss_text.items():
            losses[k+"_text"]=v

        query_embs=text_embs
        query_pids=text_pids
        return losses,query_embs,query_pids
    def forward_query(self, image_list, gt_instances):
        # one sentece for each query
        text_tokens=[]
        for inst in gt_instances:
            if hasattr(inst,"descriptions"):
                text_tokens.append(inst.descriptions[0][0][0])
        if len(text_tokens)==0:
            text_tokens=None
        else:
            text_tokens=torch.stack(text_tokens).to(self.device)
        qimgs=None
        query_embs=self.query_embeds(text_tokens,qimgs)
        query_embs = torch.split(query_embs, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=query_embs[i])
            for i in range(len(query_embs))
        ]

@META_ARCH_REGISTRY.register()
class MQueryGrounding8DeformImage(MQueryGrounding8Deform):
    def query_train(self,text_pids,qimg_pids,text_tokens,query_imgs,qimg_paired_text_tokens):
        losses={}
        # image query
        qimg_embs=self.query_embeds(None,query_imgs)
        reid_loss_qimg = self.oim_loss_fuse(qimg_embs,qimg_pids )
        for k,v in reid_loss_qimg.items():
            losses[k+"_qimg"]=v
        query_embs=qimg_embs
        query_pids=qimg_pids
        return losses,query_embs,query_pids
    def forward_query(self, image_list, gt_instances):
        # one sentece for each query
        qimgs=[]
        for inst in gt_instances:
            if hasattr(inst,"queries"):
                qimgs.append(inst.queries[0][0])
        text_tokens=None
        qimgs=torch.stack(qimgs).to(self.device)
        query_embs=self.query_embeds(text_tokens,qimgs)
        query_embs = torch.split(query_embs, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=query_embs[i])
            for i in range(len(query_embs))
        ]

@META_ARCH_REGISTRY.register()
class MQueryGrounding8DeformSplitCLS(MQueryGrounding8Deform):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        qh,qw=kws["qimg_size"]
        scale = self.clip_model.token_embedding.weight.shape[1] ** -0.5 
        self.qimg_pos_embed=nn.Parameter(scale * torch.randn( qh//16*qw//16, self.clip_model.token_embedding.weight.shape[1]))
        self.query_cls_token = nn.Parameter(scale * torch.randn(1, self.clip_model.token_embedding.weight.shape[1]))
        self.query_cls_pos = nn.Parameter(scale * torch.randn(1, self.clip_model.token_embedding.weight.shape[1]))
        self.qimg_cls_token = nn.Parameter(scale * torch.randn(1, self.clip_model.token_embedding.weight.shape[1]))
        self.qimg_cls_pos = nn.Parameter(scale * torch.randn(1, self.clip_model.token_embedding.weight.shape[1]))
    def query_embeds(self,text,img):
        # text_feats=ckpt.checkpoint(self.clip_model.encode_text,text)
        if img is None:
            q_feat=self.clip_model.token_embedding(text)+self.clip_model.positional_embedding
            q_feat = q_feat.permute(1, 0, 2)  # NLD -> LND
            mask=torch.empty(self.clip_model.context_length, self.clip_model.context_length).fill_(float("-inf")).triu_(1).to(self.device)
            q_feat = self.clip_model.transformer(q_feat,True,mask)
            q_feat = q_feat.permute(1, 0, 2)  # LND -> NLD
            q_feat = self.clip_model.ln_final(q_feat).type(self.clip_model.dtype)
            q_feat = q_feat @ self.clip_model.text_projection
            q_feat = q_feat[torch.arange(q_feat.shape[0]), text.argmax(dim=-1)]
            q_feat =self.bn_neck_text(q_feat)
        elif text is None:
            q_feat=self.qimg_tokenizer(img).flatten(2,3).transpose(1,2)+self.qimg_pos_embed.unsqueeze(0)
            q_cls=(self.qimg_cls_token+self.qimg_cls_pos).unsqueeze(0).expand(q_feat.shape[0],-1,-1)
            q_feat=torch.cat([q_cls,q_feat],dim=1)
            q_feat = q_feat.permute(1, 0, 2)  # NLD -> LND
            mask=torch.zeros((q_feat.shape[0],q_feat.shape[0]),dtype=q_feat.dtype,device=self.device)
            q_feat = self.clip_model.transformer(q_feat,True,mask)
            q_feat = q_feat.permute(1, 0, 2)  # LND -> NLD
            q_feat = self.clip_model.ln_final(q_feat).type(self.clip_model.dtype)
            q_feat = q_feat @ self.clip_model.text_projection
            q_feat = q_feat[:,0]
            q_feat = self.bn_neck_query(q_feat)
        else:
            q_feat_t=self.clip_model.token_embedding(text)+self.clip_model.positional_embedding
            q_feat_i=self.qimg_tokenizer(img).flatten(2,3).transpose(1,2)+self.qimg_pos_embed.unsqueeze(0)
            q_cls=(self.query_cls_token+self.query_cls_pos).unsqueeze(0).expand(q_feat_t.shape[0],-1,-1)
            q_feat=torch.cat([q_cls,q_feat_i,q_feat_t],dim=1)
            q_feat = q_feat.permute(1, 0, 2)  # NLD -> LND
            
            mask=torch.zeros((q_feat.shape[0],q_feat.shape[0]),dtype=q_feat.dtype,device=self.device)
            mask[-text.shape[1]:,-text.shape[1]:]=torch.empty(self.clip_model.context_length, self.clip_model.context_length).fill_(float("-inf")).triu_(1).to(self.device)
            text_idx_offset=1+self.qimg_pos_embed.shape[0]
            text_cls_idxs=text.argmax(dim=-1)
            n_heads=self.clip_model.ln_final.weight.shape[0]//64
            n_query=q_feat.shape[1]
            batched_mask=mask.repeat(n_query,1,1)
            for bi,idx in enumerate(text_cls_idxs): # mask padded text tokens
                batched_mask[bi,:,text_idx_offset+idx+1:]=float("-inf")
            batched_mask=batched_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)

            q_feat = self.clip_model.transformer(q_feat,True,batched_mask)
            q_feat = q_feat.permute(1, 0, 2)  # LND -> NLD
            q_feat = self.clip_model.ln_final(q_feat).type(self.clip_model.dtype)
            q_feat = q_feat @ self.clip_model.text_projection
            q_feat = q_feat[:,0]
            q_feat = self.bn_neck_query(q_feat)
        return q_feat

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
        self.bbox_embed=None # assign by parent module
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

    return loss.sum() / num_boxes

def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio