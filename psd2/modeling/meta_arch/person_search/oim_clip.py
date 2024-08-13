import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Optional
import logging
from ..build import META_ARCH_REGISTRY
from psd2.utils.events import get_event_storage
from psd2.config.config import configurable

from psd2.structures import Boxes, Instances, ImageList
from psd2.layers import ShapeSpec
from psd2.utils.events import get_event_storage
import copy
import torch.utils.checkpoint as ckpt
import math
import random
import cv2
from psd2.utils.visualizer import mlvl_pca_feat

from .oim_clip_v1 import OimClipSimpleBi,ClipRes5ROIHeadsPs,OimClipSimpleBiMIML2DFullyPredVe,SupConLoss,trunc_normal_

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class OimClipSimpleOimshare(OimClipSimpleBi):
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
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)


@META_ARCH_REGISTRY.register()
class OimClipSimpleOimshareObi(OimClipSimpleBi):
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
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
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
class OimClipSimpleIdshare(OimClipSimpleBi):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        id_num,feat_dim =self.oim_loss.lb_layer.lookup_table.shape
        self.classifier = nn.Linear(feat_dim, id_num)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)
    def id_loss(self,pfeats,pids):
        logits=self.classifier(pfeats)
        id_loss=F.cross_entropy(logits,pids,ignore_index=-1)
        return id_loss
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
            reid_loss = self.id_loss(box_embs, assign_ids)
            losses["loss_id_i"]=reid_loss*0.5
            # text oim
            text_tokens,text_pids=[],[]
            for inst in gt_instances:
                for texts,pid in zip(inst.descriptions,inst.gt_pids):
                    if pid >-1:
                        num_text=len(texts)
                        text_tokens.extend(texts)
                        text_pids.extend([pid]*num_text)
            if len(text_pids)==0:
                losses["loss_id_t"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.id_loss(text_embs,text_pids )
                losses["loss_id_t"]=reid_loss_text*0.5
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleOimshareSdm(OimClipSimpleIdshare):
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
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            img_embs=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_embs_per_img=torch.split(box_embs,num_ps_per_img)
            for roi_embs,roi_ids,img_gt in zip(box_embs_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_embs=roi_embs[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_embs.shape[0]<num_desc:
                            id_embs=id_embs.unsqueeze(0).repeat(math.ceil(num_desc/id_embs.shape[0]),1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_embs.shape[0])[:num_desc]
                        img_embs.append(id_embs[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(text_pids)==0:
                losses["loss_id_t"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleOimshareSdmUlbrd(OimClipSimpleOimshareSdm):
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
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            img_embs=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_embs_per_img=torch.split(box_embs,num_ps_per_img)
            for roi_embs,roi_ids,img_gt in zip(box_embs_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_embs=roi_embs[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_embs.shape[0]<num_desc:
                            id_embs=id_embs.unsqueeze(0).repeat(math.ceil(num_desc/id_embs.shape[0]),1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_embs.shape[0])[:num_desc]
                        img_embs.append(id_embs[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(text_pids)==0:
                losses["loss_id_t"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)
                ulb_box_embs=box_embs[assign_ids==-1]
                if ulb_box_embs.shape[0]==0:
                    losses["loss_ulbrd"]=torch.tensor(0.,device=self.device)
                else:
                    ulb_box_embs=F.normalize(ulb_box_embs,dim=-1)
                    paired_lb_box_embs=[]
                    re_text_embs=[]
                    for pid in torch.unique(text_pids):
                        pid_box_embs=box_embs[assign_ids==pid]
                        pid_text_embs=text_embs[text_pids==pid]
                        nv=pid_box_embs.shape[0]
                        nt=pid_text_embs.shape[0]
                        if nv<nt:
                            sample_idx=list(range(nv))+random.sample(list(range(nv)),nt-nv)
                            paired_lb_box_embs.append(pid_box_embs[sample_idx])
                        elif nv>nt:
                            sample_idx=random.sample(list(range(nv)),nt)
                            paired_lb_box_embs.append(pid_box_embs[sample_idx])
                        else:
                            paired_lb_box_embs.append(pid_box_embs)
                        re_text_embs.append(pid_text_embs)
                    paired_lb_box_embs=F.normalize(torch.cat(paired_lb_box_embs,dim=0),dim=-1)
                    re_text_embs=F.normalize(torch.cat(re_text_embs,dim=0),dim=-1)
                    v2ulb_sim=paired_lb_box_embs.mm(ulb_box_embs.t())
                    t2ulb_sim=re_text_embs.mm(ulb_box_embs.t())
                    losses["loss_ulbrd"]=F.smooth_l1_loss(v2ulb_sim,t2ulb_sim)
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)
@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugOimshareSdm(OimClipSimpleOimshareSdm):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        box_aug = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        clip_model=ret["clip_model"]
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":16}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        prev_head=ret["roi_heads"]
        res5=copy.deepcopy(prev_head.res5)
        head = ClipRes5ROIHeadsPsBoxAug(cfg, res5,box_aug,res_output_shape)
        head.attnpool=clip_model.visual.attnpool
        ret["roi_heads"]=head
        return ret

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugAttmaskOimshareSdm(OimClipSimpleDetboxaugOimshareSdm):
    def img_embes(self,roi_feats):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_prob=torch.rand(all_tokens.shape[0])
        mask_prob = torch.rand((all_tokens.shape[0],all_tokens.shape[1]+1),device=all_tokens.device) # B x L
        mask_prob=mask_prob.unsqueeze(1).repeat(1,all_tokens.shape[1]+1,1) # B x L x L
        for i,mp in enumerate(do_mask_prob):
            if mp<0.5:
                mask_ratio=torch.rand(1)*(0.4-0.04)+0.04 # refer to random erasing
                attn_mask[i][mask_prob[i]<mask_ratio[0]]=float("-inf")

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
                attn_mask=attn_mask_head
            )
            return x
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
        return embs

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttmixOimshareSdm(OimClipSimpleDetboxaugOimshareSdm):
    def _contextmix(self,roi_tokens,attn):
        perm_tokens=roi_tokens[torch.randperm(roi_tokens.shape[0])]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.randint(1,int(num_tokens*0.4),(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
    def img_embes(self,roi_feats,roi_ids_uni_ulb):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        do_mix_idxs=[]
        roi_ids_uni=torch.unique(roi_ids_uni_ulb)

        for roi_id in roi_ids_uni:
            roi_idxs= torch.nonzero(roi_ids_uni_ulb==roi_id).squeeze(1)
            n_roi=roi_idxs.shape[0]
            roi_idxs_perm=roi_idxs[torch.randperm(n_roi)]
            num_anchor=max(n_roi//2,1)
            num_mix=n_roi-num_anchor
            mix_idxs=roi_idxs_perm[:num_mix]
            do_mix_idxs.append(mix_idxs)
        do_mix_idxs=torch.cat(do_mix_idxs)

        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        # mix
        mixed_tokens=self._contextmix(all_tokens[do_mix_idxs],x_attn[do_mix_idxs])
        all_tokens[do_mix_idxs]=mixed_tokens

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
            )
            return x

        
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
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
            next_pse_ulbs=1e5
            pse_gt_pids=[]
            for inst in gt_instances:
                pse_gt_pids_i=copy.deepcopy(inst.gt_pids)
                for i,pid in enumerate(inst.gt_pids):
                    if pid==-1:
                        pse_gt_pids_i[i]=next_pse_ulbs
                        next_pse_ulbs+=1
                pse_gt_pids.append(pse_gt_pids_i)

            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                pse_gt_pids, # [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features,assign_ids)
            box_embs = self.bn_neck(box_embs)
            assign_ids[assign_ids>=1e5]=-1
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            img_embs=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_embs_per_img=torch.split(box_embs,num_ps_per_img)
            for roi_embs,roi_ids,img_gt in zip(box_embs_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_embs=roi_embs[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_embs.shape[0]<num_desc:
                            id_embs=id_embs.unsqueeze(0).repeat(math.ceil(num_desc/id_embs.shape[0]),1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_embs.shape[0])[:num_desc]
                        img_embs.append(id_embs[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(text_pids)==0:
                losses["loss_oim_t"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)

            for i, instances_i in enumerate(pred_instances):
                ids_i=assign_ids_per_img[i]
                ids_i[ids_i>=1e5]=-1
                instances_i.assign_ids = ids_i

            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)



from scipy.optimize import linear_sum_assignment
@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttoptmixOimshareSdm(OimClipSimpleDetboxaugCoAttmixOimshareSdm):
    def _contextmix(self,roi_tokens,attn):
        attn_cost=((attn.unsqueeze(1)-attn.unsqueeze(0))**2).sum(-1)
        dig_idxs=list(range(attn.shape[0]))
        attn_cost[dig_idxs,dig_idxs]=1e10
        match_row,match_col=linear_sum_assignment(attn_cost.cpu().numpy())
        perm_tokens=roi_tokens[match_col]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.randint(1,int(num_tokens*0.4),(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttoptxidmixOimshareSdm(OimClipSimpleDetboxaugCoAttoptmixOimshareSdm):
    def _contextmix(self,roi_tokens,attn,roi_ids_uni_ulb):
        attn_cost=((attn.unsqueeze(1)-attn.unsqueeze(0))**2).sum(-1)
        attn_cost[roi_ids_uni_ulb.unsqueeze(1)==roi_ids_uni_ulb.unsqueeze(0)]=1e10
        match_row,match_col=linear_sum_assignment(attn_cost.cpu().numpy())
        perm_tokens=roi_tokens[match_col]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.randint(1,int(num_tokens*0.4),(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
    def img_embes(self,roi_feats,roi_ids_uni_ulb):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        do_mix_idxs=[]
        roi_ids_uni=torch.unique(roi_ids_uni_ulb)

        for roi_id in roi_ids_uni:
            roi_idxs= torch.nonzero(roi_ids_uni_ulb==roi_id).squeeze(1)
            n_roi=roi_idxs.shape[0]
            roi_idxs_perm=roi_idxs[torch.randperm(n_roi)]
            num_anchor=max(n_roi//2,1)
            num_mix=n_roi-num_anchor
            mix_idxs=roi_idxs_perm[:num_mix]
            do_mix_idxs.append(mix_idxs)
        do_mix_idxs=torch.cat(do_mix_idxs)

        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        # mix
        mixed_tokens=self._contextmix(all_tokens[do_mix_idxs],x_attn[do_mix_idxs],roi_ids_uni_ulb[do_mix_idxs])
        all_tokens[do_mix_idxs]=mixed_tokens

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
            )
            return x

        
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
        return embs
@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttoptfixmixOimshareSdm(OimClipSimpleDetboxaugCoAttmixOimshareSdm):
    def _contextmix(self,roi_tokens,attn):
        attn_cost=((attn.unsqueeze(1)-attn.unsqueeze(0))**2).sum(-1)
        dig_idxs=list(range(attn.shape[0]))
        attn_cost[dig_idxs,dig_idxs]=1e10
        match_row,match_col=linear_sum_assignment(attn_cost.cpu().numpy())
        perm_tokens=roi_tokens[match_col]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.ones((roi_tokens.shape[0],),dtype=torch.int,device=perm_tokens.device)*int(num_tokens*0.5)
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttfixmixOimshareSdm(OimClipSimpleDetboxaugCoAttmixOimshareSdm):
    def _contextmix(self,roi_tokens,attn):
        perm_tokens=roi_tokens[torch.randperm(roi_tokens.shape[0])]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.ones((roi_tokens.shape[0],),dtype=torch.int,device=perm_tokens.device)*int(num_tokens*0.5)
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
    

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttmaskhighOimshareSdm(OimClipSimpleDetboxaugOimshareSdm):
    def img_embes(self,roi_feats,roi_ids_uni_ulb):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_prob=torch.zeros(all_tokens.shape[0])
        roi_ids_uni=torch.unique(roi_ids_uni_ulb)
        for roi_id in roi_ids_uni:
            roi_idxs= torch.nonzero(roi_ids_uni_ulb==roi_id).squeeze(1)
            mask_idxs=roi_idxs[torch.randperm(roi_idxs.shape[0])[:roi_idxs.shape[0]//2]]
            do_mask_prob[mask_idxs]=1.

        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        sort_attn_idxs=torch.argsort(x_attn,dim=-1)
        num_tokens=all_tokens.shape[1]
        num_mask_tokens=torch.randint(1,int(num_tokens*0.3),(all_tokens.shape[0],))
        t_attn_value=x_attn[list(range(all_tokens.shape[0])),sort_attn_idxs[list(range(all_tokens.shape[0])),(-num_mask_tokens).cpu().numpy().tolist()].cpu().numpy().tolist()]
        attn_mask[...,1:][(x_attn>=t_attn_value.unsqueeze(1)).unsqueeze(1).expand(-1,attn_mask.shape[1],-1)]=float("-inf")
        for i,mp in enumerate(do_mask_prob):
            if mp==0.:
                attn_mask[i]=0.

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
                attn_mask=attn_mask_head,
            )
            return x

        
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
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
            next_pse_ulbs=1e5
            pse_gt_pids=[]
            for inst in gt_instances:
                pse_gt_pids_i=copy.deepcopy(inst.gt_pids)
                for i,pid in enumerate(inst.gt_pids):
                    if pid==-1:
                        pse_gt_pids_i[i]=next_pse_ulbs
                        next_pse_ulbs+=1
                pse_gt_pids.append(pse_gt_pids_i)

            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                pse_gt_pids, # [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features,assign_ids)
            box_embs = self.bn_neck(box_embs)
            assign_ids[assign_ids>=1e5]=-1
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # text oim
            img_embs=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in assign_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in assign_ids_per_img]
            box_embs_per_img=torch.split(box_embs,num_ps_per_img)
            for roi_embs,roi_ids,img_gt in zip(box_embs_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_embs=roi_embs[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_embs.shape[0]<num_desc:
                            id_embs=id_embs.unsqueeze(0).repeat(math.ceil(num_desc/id_embs.shape[0]),1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_embs.shape[0])[:num_desc]
                        img_embs.append(id_embs[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(text_pids)==0:
                losses["loss_id_t"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)

            for i, instances_i in enumerate(pred_instances):
                ids_i=assign_ids_per_img[i]
                ids_i[ids_i>=1e5]=-1
                instances_i.assign_ids = ids_i

            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttmaskhighCoAttmixOimshareSdm(OimClipSimpleDetboxaugCoAttmaskhighOimshareSdm):
    def _contextmix(self,roi_tokens,attn):
        perm_tokens=roi_tokens[torch.randperm(roi_tokens.shape[0])]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.randint(1,int(num_tokens*0.4),(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
    def img_embes(self,roi_feats,roi_ids_uni_ulb):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_idxs=[]
        do_mix_idxs=[]
        roi_ids_uni=torch.unique(roi_ids_uni_ulb)

        for roi_id in roi_ids_uni:
            roi_idxs= torch.nonzero(roi_ids_uni_ulb==roi_id).squeeze(1)
            n_roi=roi_idxs.shape[0]
            roi_idxs_perm=roi_idxs[torch.randperm(n_roi)]
            num_anchor=max(n_roi//3,1)
            num_mix=(n_roi-num_anchor)//2
            num_mask=n_roi-num_anchor-num_mix
            mask_idxs=roi_idxs_perm[-num_mask:]
            mix_idxs=roi_idxs_perm[:num_mix]
            do_mask_idxs.append(mask_idxs)
            do_mix_idxs.append(mix_idxs)
        do_mask_idxs=torch.cat(do_mask_idxs)
        do_mix_idxs=torch.cat(do_mix_idxs)
        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        sort_attn_idxs=torch.argsort(x_attn,dim=-1)
        num_tokens=all_tokens.shape[1]
        # mask
        num_mask_tokens=torch.randint(1,int(num_tokens*0.3),(all_tokens.shape[0],))
        t_attn_value=x_attn[list(range(all_tokens.shape[0])),sort_attn_idxs[list(range(all_tokens.shape[0])),(-num_mask_tokens).cpu().numpy().tolist()].cpu().numpy().tolist()]
        for idx in do_mask_idxs:
            attn_mask[idx,:,1:][(x_attn[idx]>=t_attn_value[idx]).unsqueeze(0).expand(attn_mask.shape[1],-1)]=float("-inf")
        # mix
        mixed_tokens=self._contextmix(all_tokens[do_mix_idxs],x_attn[do_mix_idxs])
        all_tokens[do_mix_idxs]=mixed_tokens

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
                attn_mask=attn_mask_head,
            )
            return x

        
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
        return embs

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttmaskhighCoAttoptmixOimshareSdm(OimClipSimpleDetboxaugCoAttmaskhighCoAttmixOimshareSdm):
    def _contextmix(self,roi_tokens,attn):
        attn_cost=((attn.unsqueeze(1)-attn.unsqueeze(0))**2).sum(-1)
        dig_idxs=list(range(attn.shape[0]))
        attn_cost[dig_idxs,dig_idxs]=1e10
        match_row,match_col=linear_sum_assignment(attn_cost.cpu().numpy())
        perm_tokens=roi_tokens[match_col]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.randint(1,int(num_tokens*0.4),(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugAttmaskhighOimshareSdm(OimClipSimpleDetboxaugAttmaskOimshareSdm):
    def img_embes(self,roi_feats):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_prob=torch.rand(all_tokens.shape[0])

        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        sort_attn_idxs=torch.argsort(x_attn,dim=-1)
        num_tokens=all_tokens.shape[1]
        num_mask_tokens=torch.randint(1,int(num_tokens*0.3),(all_tokens.shape[0],))
        t_attn_value=x_attn[list(range(all_tokens.shape[0])),sort_attn_idxs[list(range(all_tokens.shape[0])),(-num_mask_tokens).cpu().numpy().tolist()].cpu().numpy().tolist()]
        attn_mask[...,1:][(x_attn>=t_attn_value.unsqueeze(1)).unsqueeze(1).expand(-1,attn_mask.shape[1],-1)]=float("-inf")
        for i,mp in enumerate(do_mask_prob):
            if mp>=0.5:
                attn_mask[i]=0.

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
                attn_mask=attn_mask_head,
            )
            return x

        
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
        return embs

@META_ARCH_REGISTRY.register()
class OimClipSimpleOimshareSdmMIML2DFullyPredVe(OimClipSimpleBiMIML2DFullyPredVe):
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
            box_embs,_ = self.img_embes(box_features)
            box_feats=box_features.flatten(2,3).permute(2,0,1).contiguous()
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
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_feats=img_rois[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_feats.shape[0]<num_desc:
                            id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                        img_features.append(id_feats[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(img_features)==0:
                losses["loss_mim"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugOimshareSdmMIML2DFullyPredVe(OimClipSimpleOimshareSdmMIML2DFullyPredVe):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        box_aug = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        clip_model=ret["clip_model"]
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":16}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        prev_head=ret["roi_heads"]
        res5=copy.deepcopy(prev_head.res5)
        head = ClipRes5ROIHeadsPsBoxAug(cfg, res5,box_aug,res_output_shape)
        head.attnpool=clip_model.visual.attnpool
        ret["roi_heads"]=head
        return ret

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugOimshareSdmMIML2DFullyPredVuni(OimClipSimpleDetboxaugOimshareSdmMIML2DFullyPredVe):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleDetboxaugOimshareSdmMIML2DFullyPredVuni,self).__init__(*args, **kws)
        embed_dim=self.clip_model.text_projection.shape[1]*2
        self.mim_norm=nn.LayerNorm(embed_dim//2)
        self.mim_head=nn.Linear(embed_dim//2,embed_dim//2)
        self.mim_token=nn.Parameter(torch.zeros(1,1, embed_dim))
        trunc_normal_(self.mim_token, mean=0., std=.02)
    def mlm_loss(self,i_features,tokens):
        masked_i_features=self.random_masked_tokens_and_labels(i_features)
        text_feats = self.clip_model.encode_text(torch.stack(tokens).to(self.device),ckpt=True)
        rec_i_features =self.cross_former(masked_i_features, text_feats, text_feats,with_ckpt=True)
        rec_i_features =self.mim_head(self.mim_norm(rec_i_features))
        with torch.no_grad():
            tgt_i_features=i_features.transpose(0,1)
            tgt_i_features = torch.cat([tgt_i_features.mean(dim=0, keepdim=True), tgt_i_features], dim=0)  # (HW+1)NC
            tgt_i_features = tgt_i_features + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(tgt_i_features.dtype)  # (HW+1)NC
            tgt_i_features, _ = F.multi_head_attention_forward(
                query=tgt_i_features, key=tgt_i_features, value=tgt_i_features,
                embed_dim_to_check=tgt_i_features.shape[-1],
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
            tgt_i_features=tgt_i_features.transpose(0,1)
        l2_dist=F.mse_loss(rec_i_features,tgt_i_features,reduction="mean")
        storage = get_event_storage()
        if storage.iter % self.vis_period == 0:
            h,w=8,4
            B=i_features.shape[0]
            with torch.no_grad():
                org_feats=tgt_i_features[:,1:].permute(0,2,1).reshape(B,-1,h,w)
                masked_feats=masked_i_features[:,1:].permute(0,2,1).reshape(B,-1,h,w)
                rec_feats=rec_i_features[:,1:].permute(0,2,1).reshape(B,-1,h,w)
                pca_feats=mlvl_pca_feat([org_feats,masked_feats,rec_feats])
                # vis_pca_feats=[F.interpolate(feats/255,(th,tw),mode="bilinear") for feats in pca_feats]
                vis_pca_feats=[feats/255 for feats in pca_feats]
                for i in range(B):
                    storage.put_image("mim/feat_{}_org".format(i), vis_pca_feats[0][i])
                    storage.put_image("mim/feat_{}_mask".format(i), vis_pca_feats[1][i])
                    storage.put_image("mim/feat_{}_rec".format(i), vis_pca_feats[2][i])
        return {"loss_mim":l2_dist}
    def random_masked_tokens_and_labels(self,all_tokens): # before attention pooling
        prob = torch.rand(all_tokens.shape[:2],device=all_tokens.device).unsqueeze(2) # B x L x 1
        masking_w=torch.zeros_like(prob)
        masking_w[prob<self.mim_ratio]=1
        mim_tokens=self.mim_token.expand(all_tokens.shape[0],all_tokens.shape[1],-1)
        masked_tokens=all_tokens*(1-masking_w)+mim_tokens*masking_w
        def inner_forward(x):
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
        masked_feats=ckpt.checkpoint(inner_forward,masked_tokens.transpose(0,1))
        return masked_feats.transpose(0,1)



@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttmaskhighOimshareMIML2DFullyPredVe(OimClipSimpleDetboxaugOimshareSdmMIML2DFullyPredVe):
    def img_embes(self,roi_feats,roi_ids_uni_ulb):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_prob=torch.zeros(all_tokens.shape[0])
        roi_ids_uni=torch.unique(roi_ids_uni_ulb)
        for roi_id in roi_ids_uni:
            roi_idxs= torch.nonzero(roi_ids_uni_ulb==roi_id).squeeze(1)
            mask_idxs=roi_idxs[torch.randperm(roi_idxs.shape[0])[:roi_idxs.shape[0]//2]]
            do_mask_prob[mask_idxs]=1.

        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        sort_attn_idxs=torch.argsort(x_attn,dim=-1)
        num_tokens=all_tokens.shape[1]
        num_mask_tokens=torch.randint(1,int(num_tokens*0.3),(all_tokens.shape[0],))
        t_attn_value=x_attn[list(range(all_tokens.shape[0])),sort_attn_idxs[list(range(all_tokens.shape[0])),(-num_mask_tokens).cpu().numpy().tolist()].cpu().numpy().tolist()]
        attn_mask[...,1:][(x_attn>=t_attn_value.unsqueeze(1)).unsqueeze(1).expand(-1,attn_mask.shape[1],-1)]=float("-inf")
        for i,mp in enumerate(do_mask_prob):
            if mp==0.:
                attn_mask[i]=0.

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
                attn_mask=attn_mask_head,
            )
            return x

        
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
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

            next_pse_ulbs=1e5
            pse_gt_pids=[]
            for inst in gt_instances:
                pse_gt_pids_i=copy.deepcopy(inst.gt_pids)
                for i,pid in enumerate(inst.gt_pids):
                    if pid==-1:
                        pse_gt_pids_i[i]=next_pse_ulbs
                        next_pse_ulbs+=1
                pse_gt_pids.append(pse_gt_pids_i)

            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                pse_gt_pids, # [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            assign_ids = torch.cat(assign_ids_per_img)
            pos_mask=assign_ids>-2
            box_features=box_features[pos_mask]
            assign_ids=assign_ids[pos_mask]
            box_embs = self.img_embes(box_features,assign_ids)
            assign_ids[assign_ids>=1e5]=-1
            box_feats=box_features.flatten(2,3).permute(2,0,1).contiguous()
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
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_feats=img_rois[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_feats.shape[0]<num_desc:
                            id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                        img_features.append(id_feats[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(img_features)==0:
                losses["loss_mim"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[assign_ids>-1]
                lb_assign_ids=assign_ids[assign_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)
            for i, instances_i in enumerate(pred_instances):
                ids_i=assign_ids_per_img[i]
                ids_i[ids_i>=1e5]=-1
                instances_i.assign_ids = ids_i
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugCoAttmaskhighCoattmixOimshareMIML2DFullyPredVe(OimClipSimpleDetboxaugCoAttmaskhighOimshareMIML2DFullyPredVe):
    def _contextmix(self,roi_tokens,attn):
        perm_tokens=roi_tokens[torch.randperm(roi_tokens.shape[0])]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.randint(1,int(num_tokens*0.4),(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
    def img_embes(self,roi_feats,roi_ids_uni_ulb):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_idxs=[]
        do_mix_idxs=[]
        roi_ids_uni=torch.unique(roi_ids_uni_ulb)

        for roi_id in roi_ids_uni:
            roi_idxs= torch.nonzero(roi_ids_uni_ulb==roi_id).squeeze(1)
            n_roi=roi_idxs.shape[0]
            roi_idxs_perm=roi_idxs[torch.randperm(n_roi)]
            num_anchor=max(n_roi//3,1)
            num_mix=(n_roi-num_anchor)//2
            num_mask=n_roi-num_anchor-num_mix
            mask_idxs=roi_idxs_perm[-num_mask:]
            mix_idxs=roi_idxs_perm[:num_mix]
            do_mask_idxs.append(mask_idxs)
            do_mix_idxs.append(mix_idxs)
        do_mask_idxs=torch.cat(do_mask_idxs)
        do_mix_idxs=torch.cat(do_mix_idxs)
        with torch.no_grad():
            x=all_tokens
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, x_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            x_attn=x_attn[:,0,1:]
        sort_attn_idxs=torch.argsort(x_attn,dim=-1)
        num_tokens=all_tokens.shape[1]
        # mask
        num_mask_tokens=torch.randint(1,int(num_tokens*0.3),(all_tokens.shape[0],))
        t_attn_value=x_attn[list(range(all_tokens.shape[0])),sort_attn_idxs[list(range(all_tokens.shape[0])),(-num_mask_tokens).cpu().numpy().tolist()].cpu().numpy().tolist()]
        for idx in do_mask_idxs:
            attn_mask[idx,:,1:][(x_attn[idx]>=t_attn_value[idx]).unsqueeze(0).expand(attn_mask.shape[1],-1)]=float("-inf")
        # mix
        mixed_tokens=self._contextmix(all_tokens[do_mix_idxs],x_attn[do_mix_idxs])
        all_tokens[do_mix_idxs]=mixed_tokens

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
                attn_mask=attn_mask_head,
            )
            return x

        
        img_feats=inner_forward(all_tokens)
        embs=img_feats[0]
        return embs

@META_ARCH_REGISTRY.register()
class OimClipSimpleMaskOimshareSdmMIML2DFullyPredVe(OimClipSimpleOimshareSdmMIML2DFullyPredVe):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ) -> None:
        super(OimClipSimpleMaskOimshareSdmMIML2DFullyPredVe,self).__init__(*args, **kws)
        self.vmask_token=copy.deepcopy(self.mim_token)
    def img_embes(self,roi_feats):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        if torch.rand(1)[0]<0.5:
            prob = torch.rand(all_tokens.shape[:2],device=all_tokens.device).unsqueeze(2) # B x L x 1
            masking_w=torch.zeros_like(prob)
            mask_ratio=torch.rand(1)*(0.4-0.02)+0.02 # refer to random erasing
            masking_w[prob<mask_ratio[0]]=1
            mim_tokens=self.mim_token.expand(all_tokens.shape[0],all_tokens.shape[1],-1)
            all_tokens=all_tokens*(1-masking_w)+mim_tokens*masking_w
        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
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
        img_feats=inner_forward(all_tokens)
        feats=img_feats[1:]
        embs=feats.mean(dim=0)
        if self.training:
            return embs,feats
        else:
            return embs

from psd2.modeling.box_augmentation import build_box_augmentor
@META_ARCH_REGISTRY.register()
class OimClipSimpleBoxaugOimshareSdmMIML2DFullyPredVe(OimClipSimpleOimshareSdmMIML2DFullyPredVe):
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
            # box aug
            pos_boxes, pos_ids = [], []
            for gts_i in gt_instances:
                # append gt
                pos_boxes.append(gts_i.gt_boxes.tensor)
                pos_ids.append(gts_i.gt_pids)
            pos_boxes_per_img, pos_ids_per_img = self.box_aug.augment_boxes(
                pos_boxes,
                pos_ids,
                det_boxes=None,
                det_pids=None,
                img_sizes=[gti.image_size for gti in gt_instances],
            )
            pos_box_features=self.roi_heads._shared_roi_transform(
                [features[f] for f in self.roi_heads.in_features], 
                [Boxes(boxes_i) for boxes_i in pos_boxes_per_img]
            )
            pos_ids=torch.cat(pos_ids_per_img)
            box_embs,_ = self.img_embes(pos_box_features)
            box_feats=pos_box_features.flatten(2,3).permute(2,0,1).contiguous()
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, pos_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in pos_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in pos_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_feats=img_rois[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_feats.shape[0]<num_desc:
                            id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                        img_features.append(id_feats[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(img_features)==0:
                losses["loss_mim"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[pos_ids>-1]
                lb_assign_ids=pos_ids[pos_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleBoxaugAttdropOimshareSdmMIML2DFullyPredVe(OimClipSimpleBoxaugOimshareSdmMIML2DFullyPredVe):
    @configurable
    def __init__(self, vis_attn_drop, *args, **kws):
        super().__init__(*args, **kws)
        self.vis_attn_drop = vis_attn_drop
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["vis_attn_drop"] = cfg.PERSON_SEARCH.DET.CLIP.VIS_ATTN_DROPOUT
        return ret
    def img_embes(self,roi_feats):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        def inner_forward(x):
            if torch.rand(1)[0]<0.5:
                attn_drop=self.vis_attn_drop
            else:
                attn_drop=0.
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
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
                dropout_p=attn_drop,
                out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
                out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.clip_model.visual.attnpool.training,
                need_weights=False
            )
            return x
        img_feats=inner_forward(all_tokens)
        feats=img_feats[1:]
        embs=feats.mean(dim=0)
        if self.training:
            return embs,feats
        else:
            return embs

@META_ARCH_REGISTRY.register()
class OimClipSimpleBoxaugAttmaskOimshareSdmMIML2DFullyPredVe(OimClipSimpleBoxaugOimshareSdmMIML2DFullyPredVe):
    def img_embes(self,roi_feats):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_prob=torch.rand(all_tokens.shape[0])
        mask_prob = torch.rand((all_tokens.shape[0],all_tokens.shape[1]+1),device=all_tokens.device) # B x L
        mask_prob=mask_prob.unsqueeze(1).repeat(1,all_tokens.shape[1]+1,1) # B x L x L
        for i,mp in enumerate(do_mask_prob):
            if mp<0.5:
                mask_ratio=torch.rand(1)*(0.4-0.02)+0.02 # refer to random erasing
                attn_mask[i][mask_prob[i]<mask_ratio[0]]=float("-inf")

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=False,
                attn_mask=attn_mask_head
            )
            return x
        img_feats=inner_forward(all_tokens)
        feats=img_feats[1:]
        embs=feats.mean(dim=0)
        if self.training:
            return embs,feats
        else:
            return embs

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugAttmaskOimshareSdmMIML2DFullyPredVe(OimClipSimpleBoxaugAttmaskOimshareSdmMIML2DFullyPredVe):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        box_aug = build_box_augmentor(cfg.PERSON_SEARCH.REID.BOX_AUGMENTATION)
        clip_model=ret["clip_model"]
        out_feature_strides = {"res2":4,"res3":8,"res4":16,"res5":16}
        out_feature_channels = {"res2":256,"res3":512,"res4":1024,"res5":2048}
        res_output_shape={
            name: ShapeSpec(
                channels=out_feature_channels[name],
                stride=out_feature_strides[name],
            )
            for name in ["res2","res3","res4","res5"]
        }
        prev_head=ret["roi_heads"]
        res5=copy.deepcopy(prev_head.res5)
        head = ClipRes5ROIHeadsPsBoxAug(cfg, res5,box_aug,res_output_shape)
        head.attnpool=clip_model.visual.attnpool
        ret["roi_heads"]=head
        return ret
    def img_embes(self,roi_feats):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_prob=torch.rand(all_tokens.shape[0])
        mask_prob = torch.rand((all_tokens.shape[0],all_tokens.shape[1]+1),device=all_tokens.device) # B x L
        mask_prob=mask_prob.unsqueeze(1).repeat(1,all_tokens.shape[1]+1,1) # B x L x L
        for i,mp in enumerate(do_mask_prob):
            if mp<0.5:
                mask_ratio=torch.rand(1)*(0.4-0.02)+0.02 # refer to random erasing
                attn_mask[i][mask_prob[i]<mask_ratio[0]]=float("-inf")

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
                attn_mask=attn_mask_head,
            )
            return x,attn
        img_feats,attn_weights=inner_forward(all_tokens)
        attn_cls=attn_weights[:,0,1:]
        feats=img_feats[1:]
        embs=feats.mean(dim=0)
        if self.training:
            return embs,feats,attn_cls.reshape(-1,roi_feats.shape[-2],roi_feats.shape[-1])
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
            # box aug
            pos_boxes, pos_ids = [], []
            for gts_i in gt_instances:
                # append gt
                pos_boxes.append(gts_i.gt_boxes.tensor)
                pos_ids.append(gts_i.gt_pids)
            pos_boxes_per_img, pos_ids_per_img = self.box_aug.augment_boxes(
                pos_boxes,
                pos_ids,
                det_boxes=None,
                det_pids=None,
                img_sizes=[gti.image_size for gti in gt_instances],
            )
            pos_box_features=self.roi_heads._shared_roi_transform(
                [features[f] for f in self.roi_heads.in_features], 
                [Boxes(boxes_i) for boxes_i in pos_boxes_per_img]
            )
            pos_ids=torch.cat(pos_ids_per_img)
            box_embs,_,attn_cls = self.img_embes(pos_box_features)
            box_feats=pos_box_features.flatten(2,3).permute(2,0,1).contiguous()
            box_embs = self.bn_neck(box_embs)
            reid_loss = self.oim_loss(box_embs, pos_ids)
            for k,v in reid_loss.items():
                losses[k]=v*0.5
            # NOTE vis attn
            storage = get_event_storage()
            attn_cls=attn_cls.detach()
            idx_offset=0
            feat_hw=pos_box_features.shape[-2:]
            if self.vis_period >0 and  storage.iter % self.vis_period == 0:
                for i in range(len(gt_instances)):
                    vis_imgs=[]
                    boxes=pos_boxes_per_img[i]
                    img_t = image_list.tensor[i]
                    img_rgb = (
                        img_t * self.pixel_std + self.pixel_mean
                    ).cpu() # .numpy().transpose(1, 2, 0) * 255
                    for bi,i_box in enumerate(boxes):
                        i_box=i_box.cpu().numpy().astype(np.int32)
                        i_img=img_rgb[:,i_box[1]:i_box[3], i_box[0]:i_box[2]]
                        i_attn=attn_cls[bi+idx_offset].cpu().numpy()
                        i_attn=(i_attn-i_attn.min())/(i_attn.max()-i_attn.min())*255
                        i_attn=cv2.applyColorMap(i_attn.astype(np.uint8),cv2.COLORMAP_JET)[...,::-1]
                        i_img=F.interpolate(i_img[None],(feat_hw[0]*32,feat_hw[1]*32))[0]
                        i_attn=torch.tensor(i_attn.copy()).permute(2,0,1)/255.
                        i_attn=F.interpolate(i_attn[None],(feat_hw[0]*32,feat_hw[1]*32))[0]
                        vis_imgs.append(0.7*i_img+0.3*i_attn)
                    vis_img=torch.cat(vis_imgs,dim=-1)
                    storage.put_image("img_{}/attn".format(i), vis_img)
                    idx_offset+=boxes.shape[0]
            # NOTE randomly sample one roi feature for one text tokens
            img_features=[]
            text_tokens=[]
            text_pids=[]
            num_ps_per_img=[(ids>-2).nonzero().shape[0] for ids in pos_ids_per_img]
            roi_p_ids_per_img=[ids[ids>-2] for ids in pos_ids_per_img]
            box_features_per_img=torch.split(box_feats.permute(1,0,2).contiguous(),num_ps_per_img) # LBC -> BLC
            for img_rois,roi_ids,img_gt in zip(box_features_per_img,roi_p_ids_per_img,gt_instances):
                gt_ids=img_gt.gt_pids
                gt_tokens=img_gt.descriptions
                for pid,p_tokens in zip(gt_ids,gt_tokens):
                    if pid>-1:
                        # there can be multiple text seq for one id
                        id_feats=img_rois[roi_ids==pid]
                        num_desc=len(p_tokens)
                        if id_feats.shape[0]<num_desc:
                            id_feats=id_feats.unsqueeze(0).repeat(math.ceil(num_desc/id_feats.shape[0]),1,1,1).flatten(0,1)
                        sampled_idxs=torch.randperm(id_feats.shape[0])[:num_desc]
                        img_features.append(id_feats[sampled_idxs])
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid]*num_desc)
            if len(img_features)==0:
                losses["loss_mim"]=torch.tensor(0.,device=self.device)
            else:
                roi_feats=torch.cat(img_features)
                losses.update(self.mlm_loss(roi_feats,text_tokens))

            # text oim
            if len(text_pids)==0:
                losses["loss_oim_text"]=torch.tensor(0.,device=self.device)
                losses["loss_sdm"]=torch.tensor(0.,device=self.device)
            else:
                text_pids=torch.stack(text_pids).to(self.device)
                text_embs=self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs,text_pids )
                for k,v in reid_loss_text.items():
                    losses[k+"_text"]=v*0.5
                lb_box_embs=box_embs[pos_ids>-1]
                lb_assign_ids=pos_ids[pos_ids>-1]
                losses["loss_sdm"]=self.compute_sdm(lb_box_embs,text_embs,lb_assign_ids,text_pids,1/0.02)
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            return super().forward_gallery(image_list, gt_instances)

@META_ARCH_REGISTRY.register()
class OimClipSimpleDetboxaugAttmaskAttmixOimshareSdmMIML2DFullyPredVe(OimClipSimpleDetboxaugAttmaskOimshareSdmMIML2DFullyPredVe):
    def _contextmix(self,roi_tokens,attn):
        perm_tokens=roi_tokens[torch.randperm(roi_tokens.shape[0])]
        sort_attn_idxs=torch.argsort(attn,dim=-1)
        num_tokens=roi_tokens.shape[1]
        num_mix_tokens=torch.randint(1,int(num_tokens*0.4),(roi_tokens.shape[0],))
        keep_mask=torch.ones_like(attn)
        t_attn_value=attn[list(range(roi_tokens.shape[0])),sort_attn_idxs[list(range(roi_tokens.shape[0])),(num_mix_tokens-1).cpu().numpy().tolist()].cpu().numpy().tolist()]
        keep_mask[attn<=t_attn_value.unsqueeze(1)]=0.
        keep_mask=keep_mask.unsqueeze(2)
        roi_tokens=keep_mask*roi_tokens+(1-keep_mask)*perm_tokens
        return roi_tokens
    def img_embes(self,roi_feats):
        all_tokens=roi_feats.flatten(2,3).permute(0,2,1).contiguous() # B x L x C
        attn_mask=torch.zeros((all_tokens.shape[0],all_tokens.shape[1]+1,all_tokens.shape[1]+1),dtype=all_tokens.dtype,device=all_tokens.device)
        do_mask_prob=torch.rand(all_tokens.shape[0])
        mask_prob = torch.rand((all_tokens.shape[0],all_tokens.shape[1]+1),device=all_tokens.device) # B x L
        mask_prob=mask_prob.unsqueeze(1).repeat(1,all_tokens.shape[1]+1,1) # B x L x L
        for i,mp in enumerate(do_mask_prob):
            if mp<0.5:
                mask_ratio=torch.rand(1)*(0.4-0.02)+0.02 # refer to random erasing
                attn_mask[i][mask_prob[i]<mask_ratio[0]]=float("-inf")

        def inner_forward(x):
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            attn_mask_head=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1).flatten(0,1)
            x, attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
                attn_mask=attn_mask_head,
            )
            return x,attn
        # select attmix samples
        num_samples=all_tokens.shape[0]
        mix_num_samples=num_samples//2
        if mix_num_samples%2!=0: 
            mix_num_samples-=1
        mix_idx=torch.randperm(num_samples)[:mix_num_samples]
        with torch.no_grad():
            x=all_tokens[mix_idx]
            x = x.transpose(0,1)  # N(HW)C -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            n_heads=self.clip_model.visual.attnpool.num_heads
            _, mix_attn = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=n_heads,
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
                need_weights=True,
            )
            mix_attn=mix_attn[:,0,1:]
        mixed_tokens=self._contextmix(all_tokens[mix_idx],mix_attn)
        all_tokens[mix_idx]=mixed_tokens
        img_feats,attn_weights=inner_forward(all_tokens)
        attn_cls=attn_weights[:,0,1:]
        feats=img_feats[1:]
        embs=feats.mean(dim=0)
        if self.training:
            return embs,feats,attn_cls.reshape(-1,roi_feats.shape[-2],roi_feats.shape[-1])
        else:
            return embs

class ClipRes5ROIHeadsPsBoxAug(ClipRes5ROIHeadsPs):
    @configurable
    def __init__(
        self,
        box_aug,
        *args,
        **kwargs,
    ):
        super().__init__(*args,**kwargs)
        self.box_aug=box_aug
    @classmethod
    def from_config(cls, cfg,res5,box_aug, input_shape):
        ret = super().from_config(cfg,res5,input_shape)
        ret["box_aug"]=box_aug
        return ret
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
            """
            proposals: list of Instances with
            image_size, proposal_boxes, objectness_logits, gt_classes, gt_boxes, gt_pids
            pos_match_indices: list of
            (src_idxs, tgt_idxs)
            """
            for i,gts_i in enumerate(targets):
                gt_boxes=gts_i.gt_boxes.tensor
                match_idxs=torch.arange(gts_i.gt_pids.shape[0],device=gts_i.gt_pids.device)
                aug_boxes, aug_match_idxs = self.box_aug.augment_boxes(
                    [gt_boxes],
                    [match_idxs],
                    det_boxes=None,
                    det_pids=None,
                    img_sizes=[gts_i.image_size],
                )
                aug_boxes,aug_match_idxs=aug_boxes[0],aug_match_idxs[0]
                num_aug=aug_boxes.shape[0]
                num_prev=proposals[i].proposal_boxes.tensor.shape[0]
                proposals[i].proposal_boxes.tensor=torch.cat([proposals[i].proposal_boxes.tensor,aug_boxes])
                proposals[i].objectness_logits=torch.cat([proposals[i].objectness_logits,torch.ones(num_aug,device=aug_boxes.device)])
                proposals[i].gt_classes=torch.cat([proposals[i].gt_classes,gts_i.gt_classes[aug_match_idxs]])
                p_gt_boxes=proposals[i].gt_boxes.tensor
                aug_gt_boxes=gt_boxes[aug_match_idxs]
                proposals[i].gt_boxes.tensor=torch.cat([p_gt_boxes,aug_gt_boxes])
                proposals[i].gt_pids=torch.cat([proposals[i].gt_pids,gts_i.gt_pids[aug_match_idxs]])
                
                src_idxs=torch.cat( [pos_match_indices[i][0],torch.arange(num_aug,device=aug_boxes.device)+num_prev])
                tgt_idxs=torch.cat( [pos_match_indices[i][1],aug_match_idxs])
                pos_match_indices[i]=(src_idxs,tgt_idxs)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        
        box_embs = self.box_embedding(box_features)
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