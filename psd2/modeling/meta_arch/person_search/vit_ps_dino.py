import torch
import torch.nn as nn
from torch.nn import functional as F

from ..build import META_ARCH_REGISTRY
from psd2.config import configurable
from psd2.structures import Boxes, Instances, BoxMode
from psd2.modeling.id_assign import build_id_assigner
from psd2.utils.events import get_event_storage
from psd2.utils import comm
from psd2.layers import batched_nms
import math
import numpy as np
from functools import partial

from torch.utils import checkpoint
from copy import deepcopy
import itertools

from psd2.modeling.meta_arch.person_search.vit_ps import (
    VitDetPsParallel,
    VisionTransformer,
    trunc_normal_,
    SetCriterion,
    MLP,
)
from .base import SearchBase


class ViTDetPsWithExtTokens(VitDetPsParallel):
    def forward_features_with_tokens(
        self,
        feat_seq,
        imgt_shape,
        token_ext,
        init_pos_ext,
        mid_pos_ext=None,
        mid_pos_reid_ext=None,
    ):
        # import pdb;pdb.set_trace()
        B, H, W = imgt_shape[0], imgt_shape[2], imgt_shape[3]
        len_per_img = 2 + token_ext.shape[1]
        num_imgs = B // len_per_img
        all_det_tokens = torch.cat(
            [
                self.det_token.unsqueeze(1),
                self.det_token.clone().unsqueeze(1),
                token_ext,
            ],
            dim=1,
        )
        all_det_tokens = all_det_tokens.expand(num_imgs, -1, -1, -1).flatten(0, 1)
        all_det_token_init_pos = torch.cat(
            [
                self.pos_embed[:, -self.det_token_num :].unsqueeze(1),
                self.pos_embed[:, -self.det_token_num :].clone().unsqueeze(1),
                init_pos_ext,
            ],
            dim=1,
        )
        all_det_token_init_pos = all_det_token_init_pos.expand(
            num_imgs, -1, -1, -1
        ).flatten(0, 1)
        if self.has_mid_pe:
            all_det_token_mid_pos = torch.cat(
                [
                    self.mid_pos_embed[:, :, -self.det_token_num :].unsqueeze(2),
                    self.mid_pos_embed[:, :, -self.det_token_num :]
                    .clone()
                    .unsqueeze(2),
                    mid_pos_ext,
                ],
                dim=2,
            )
            all_det_token_mid_pos = all_det_token_mid_pos.expand(
                -1, num_imgs, -1, -1, -1
            ).flatten(1, 2)
        else:
            all_det_token_mid_pos = None
        if self.has_mid_pe_reid:
            all_det_token_mid_pos_reid = torch.cat(
                [
                    self.mid_pos_embed_reid[:, :, -self.det_token_num :].unsqueeze(2),
                    self.mid_pos_embed_reid[:, :, -self.det_token_num :]
                    .clone()
                    .unsqueeze(2),
                    mid_pos_reid_ext,
                ],
                dim=2,
            )
            all_det_token_mid_pos_reid = all_det_token_mid_pos_reid.expand(
                -1, num_imgs, -1, -1, -1
            ).flatten(1, 2)
        else:
            all_det_token_mid_pos_reid = None
        x = feat_seq
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(
                self.pos_embed, img_size=(H, W)
            )
        else:
            temp_pos_embed = self.pos_embed
        patch_pos = temp_pos_embed[:, : -self.det_token_num].expand(B, -1, -1)
        temp_pos_embed = torch.cat([patch_pos, all_det_token_init_pos], dim=1)

        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed
            patch_pos = temp_mid_pos_embed[:, :, : -self.det_token_num].expand(
                -1, B, -1, -1
            )
            temp_mid_pos_embed = torch.cat([patch_pos, all_det_token_mid_pos], dim=2)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        det_token = all_det_tokens
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(self.para_start_idx):
            # x = self.blocks[i](x)
            x = checkpoint.checkpoint(self.blocks[i], x)  # saves mem, takes time
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]
        det_x = x
        for i in range(self.para_start_idx, len(self.blocks)):
            det_x = checkpoint.checkpoint(
                self.blocks[i], det_x
            )  # saves mem, takes time
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    det_x = det_x + temp_mid_pos_embed[i]
        det_x = self.norm(det_x)
        reid_x = x
        # interpolate mid pe reid
        if self.has_mid_pe_reid:
            # temp_mid_pos_embed = []
            if (
                self.mid_pos_embed_reid.shape[2] - 1 - self.det_token_num
            ) != reid_x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed_reid, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed_reid
            patch_pos = temp_mid_pos_embed[:, :, : -self.det_token_num].expand(
                -1, B, -1, -1
            )
            temp_mid_pos_embed = torch.cat(
                [patch_pos, all_det_token_mid_pos_reid], dim=2
            ).flatten(1, 2)

        for i in range(self.depth_reid):
            reid_x = checkpoint.checkpoint(
                self.blocks_reid[i], reid_x
            )  # saves mem, takes time
            if self.has_mid_pe_reid:
                if i < (self.depth_reid - 1):
                    reid_x = reid_x + temp_mid_pos_embed[i]
        reid_x = self.norm_reid(reid_x)

        return det_x, reid_x

    def forward_features_reid(self, feat_seq, imgt_shape):
        # import pdb;pdb.set_trace()
        B, H, W = imgt_shape[0], imgt_shape[2], imgt_shape[3]
        x = feat_seq
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(
                self.pos_embed, img_size=(H, W)
            )
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(self.para_start_idx):
            # x = self.blocks[i](x)
            x = checkpoint.checkpoint(self.blocks[i], x)  # saves mem, takes time
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]
        reid_x = x
        # interpolate mid pe reid
        if self.has_mid_pe_reid:
            # temp_mid_pos_embed = []
            if (
                self.mid_pos_embed_reid.shape[2] - 1 - self.det_token_num
            ) != reid_x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed_reid, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed_reid

        for i in range(self.depth_reid):
            reid_x = checkpoint.checkpoint(
                self.blocks_reid[i], reid_x
            )  # saves mem, takes time
            if self.has_mid_pe_reid:
                if i < (self.depth_reid - 1):
                    reid_x = reid_x + temp_mid_pos_embed[i]
        reid_x = self.norm_reid(reid_x)

        return reid_x

    def forward(self, feat_seq, imgt_shape, return_attention=False, *args, **kws):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(feat_seq, imgt_shape)
        else:
            if self.training:
                x = self.forward_features_with_tokens(
                    feat_seq, imgt_shape, *args, **kws
                )
            else:
                x = self.forward_features(feat_seq, imgt_shape)
            return x


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_iters,
        niters,
        epoch_iters,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.epoch_iters = epoch_iters
        nepochs = math.ceil(niters / epoch_iters)
        warmup_teacher_temp_epochs = math.ceil(warmup_teacher_temp_iters / epoch_iters)
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, iter_idx):
        # NOTE consider batch dim
        """
        student_output / teacher_output:
            [
                [v1,i1],
                [v1,i2],...,
                [v2,i1],
                [v2,i2],...
            ]
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        epoch = iter_idx // self.epoch_iters
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)  # identity-chunk per view

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # identity-chunk per view

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.sum()  # .mean(-1)
                n_loss_terms += loss.shape[0]
        comm.synchronize()
        all_loss_terms = comm.all_gather(n_loss_terms)
        n_loss_terms = max(sum(all_loss_terms) / comm.get_world_size(), 1.0)
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        teacher_output: (sum num_global_view x num_person on all images in this batch) x num_cluster
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        num_inst = teacher_output.shape[0]
        comm.synchronize()
        all_sum_batch_center = comm.all_gather(batch_center)
        batch_center = sum([emb.to(self.center.device) for emb in all_sum_batch_center])
        all_num_inst = comm.all_gather(num_inst)
        num_inst = sum(all_num_inst)
        batch_center = batch_center / num_inst

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


@META_ARCH_REGISTRY.register()
class VitPSParaDINO(SearchBase):
    @configurable
    def __init__(
        self,
        cfg,
        transformer,
        criterion,
        num_classes,
        num_queries,
        in_features,
        id_assigner,
        bn_neck,
        do_nms,
        num_local_tkgroups,
        transformer_teacher,
        head_student,
        head_teacher,
        dino_loss,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        # vitpara
        self.transformer = transformer
        hidden_dim = transformer.hidden_dim
        self.use_focal = criterion.use_focal
        if self.use_focal:
            self.class_embed = MLP(
                hidden_dim, hidden_dim, num_classes, 3
            )  # to avoid loading coco parameters
        else:
            self.class_embed = MLP(
                hidden_dim, hidden_dim, num_classes + 1, 3
            )  # to avoid loading coco parameters
        self.bbox_embed = MLP(
            hidden_dim, hidden_dim, 4, 3
        )  # to avoid loading coco parameters
        self.criterion = criterion
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.id_assigner = id_assigner
        self.bn_neck = bn_neck
        self.in_features = in_features
        self.cfg = cfg
        self.do_nms = do_nms

        self.class_embed_ext = nn.ModuleList(
            [deepcopy(self.class_embed) for _ in range(num_local_tkgroups)]
        )
        self.bbox_embed_ext = nn.ModuleList(
            [deepcopy(self.bbox_embed) for _ in range(num_local_tkgroups)]
        )
        self.vit_teacher = transformer_teacher
        self.head_student = head_student
        self.head_teacher = head_teacher
        self.dino_loss = dino_loss
        self.num_local_token_groups = num_local_tkgroups
        self.bn_neck_teacher = deepcopy(self.bn_neck)
        self.backbone_teacher = deepcopy(self.backbone)
        for p in self.vit_teacher.parameters():
            p.requires_grad = False
        for p in self.head_teacher.parameters():
            p.requires_grad = False
        for p in self.bn_neck_teacher.parameters():
            p.requires_grad = False
        for p in self.backbone_teacher.parameters():
            p.requires_grad = False
        with torch.no_grad():
            # local tokens
            self.local_token_groups = nn.Parameter(
                torch.stack(
                    [
                        self.transformer.det_token.clone()
                        for _ in range(num_local_tkgroups)
                    ],
                    dim=1,
                )
            )
            # local token pos
            self.local_token_pos_init = nn.Parameter(
                torch.stack(
                    [
                        self.transformer.pos_embed[
                            :, -self.transformer.det_token_num :, :
                        ].clone()
                        for _ in range(num_local_tkgroups)
                    ],
                    dim=1,
                )
            )
            self.local_token_pos_mid = (
                nn.Parameter(
                    torch.stack(
                        [
                            self.transformer.mid_pos_embed[
                                :, :, -self.transformer.det_token_num :, :
                            ].clone()
                            for _ in range(num_local_tkgroups)
                        ],
                        dim=2,
                    )
                )
                if self.transformer.has_mid_pe
                else None
            )
            self.local_token_pos_mid_reid = (
                nn.Parameter(
                    torch.stack(
                        [
                            self.transformer.mid_pos_embed_reid[
                                :, :, -self.transformer.det_token_num :, :
                            ].clone()
                            for _ in range(num_local_tkgroups)
                        ],
                        dim=2,
                    )
                )
                if self.transformer.has_mid_pe_reid
                else None
            )

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = build_id_assigner(cfg)

        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        dino_cfg = cfg.PERSON_SEARCH.DINO
        reid_cfg = cfg.PERSON_SEARCH.REID
        vit = ViTDetPsWithExtTokens(
            depth_reid=reid_cfg.MODEL.TRANSFORMER.DEPTH,
            pretrain_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size[0]
            if isinstance(patch_embed.patch_size, tuple)
            else patch_embed.patch_size,
            embed_dim=patch_embed.embed_dim,
            num_patches=patch_embed.num_patches,
            depth=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            is_distill=tr_cfg.DEIT,
        )
        det_cfg = cfg.PERSON_SEARCH.DET
        vit.finetune_det(
            img_size=tr_cfg.INIT_PE_SIZE,
            det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
            mid_pe_size=tr_cfg.MID_PE_SIZE,
        )
        ret["transformer"] = vit
        ret["head_student"] = DINOHead(
            patch_embed.embed_dim,
            dino_cfg.OUT_DIM,
            dino_cfg.BN_IN_HEAD,
            dino_cfg.NORM_LAST_LAYER,
        )
        vit = ViTDetPsWithExtTokens(
            depth_reid=reid_cfg.MODEL.TRANSFORMER.DEPTH,
            pretrain_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size[0]
            if isinstance(patch_embed.patch_size, tuple)
            else patch_embed.patch_size,
            embed_dim=patch_embed.embed_dim,
            num_patches=patch_embed.num_patches,
            depth=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            is_distill=tr_cfg.DEIT,
        )
        det_cfg = cfg.PERSON_SEARCH.DET
        vit.finetune_det(
            img_size=tr_cfg.INIT_PE_SIZE,
            det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
            mid_pe_size=tr_cfg.MID_PE_SIZE,
        )
        ret["transformer_teacher"] = vit
        ret["transformer_teacher"].load_state_dict(ret["transformer"].state_dict())
        ret["head_teacher"] = DINOHead(
            patch_embed.embed_dim,
            dino_cfg.OUT_DIM,
            dino_cfg.BN_IN_HEAD,
        )
        ret["head_teacher"].load_state_dict(ret["head_student"].state_dict())
        assert cfg.SOLVER.EPOCH_ITERS > 0, "Invalid SOLVER.EPOCH_ITERS !"
        ret["dino_loss"] = DINOLoss(
            dino_cfg.OUT_DIM,
            dino_cfg.NUM_LOCAL_TOKEN_GROUPS + 2,
            dino_cfg.WARMUP_TEACHER_TEMP,
            dino_cfg.TEACHER_TEMP,
            dino_cfg.WARMUP_TEACHER_TEMP_ITERS,
            cfg.SOLVER.MAX_ITER,
            cfg.SOLVER.EPOCH_ITERS,
        )
        ret["num_local_tkgroups"] = dino_cfg.NUM_LOCAL_TOKEN_GROUPS

        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        if reid_cfg.MODEL.BN_NECK:
            bn_neck = nn.BatchNorm1d(reid_cfg.MODEL.EMB_DIM)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["do_nms"] = det_cfg.MODEL.DO_NMS
        return ret

    @torch.no_grad()
    def ema_update(self, momentum):
        for param_q, param_k in zip(
            self.vit_student.parameters(), self.vit_teacher.parameters()
        ):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
        for param_q, param_k in zip(
            self.head_student.parameters(), self.head_teacher.parameters()
        ):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
        for param_q, param_k in zip(
            self.backbone.parameters(), self.backbone_teacher.parameters()
        ):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
        for param_q, param_k in zip(
            self.bn_neck.parameters(), self.bn_neck_teacher.parameters()
        ):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def load_state_dict(self, state_dict, strict):
        output = super().load_state_dict(state_dict, strict)
        if len(output[0]) > 0:  # missing_keys
            # not resume, NOTE assume head and ext tokens are not included
            self.vit_teacher.load_state_dict(self.vit_student.state_dict())
            # TODO remove from missing_keys
        return output

    @property
    def vit_student(self):
        return self.transformer

    def forward(self, input_list):
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
            if self.training:
                input_list = list(
                    itertools.chain(
                        *[item["global"] + item["local"] for item in input_list]
                    )
                )
                gt_instances = [gti["instances"].to(self.device) for gti in input_list]
                image_list = self.preprocess_input(input_list)
                preds, feat_maps, losses = self.forward_gallery(
                    image_list, gt_instances
                )
                self.visualize_training(image_list, gt_instances, preds, feat_maps)
                return losses
            else:
                image_list = self.preprocess_input(input_list)
                gt_instances = [gti["instances"].to(self.device) for gti in input_list]
                return self.forward_gallery(image_list, gt_instances)  # preds only

    def forward_gallery(self, image_list, gt_instances):
        if not self.training:
            return super().forward_gallery(image_list, gt_instances)
        # NOTE 1. modify local-view annotaions
        len_per_img = 2 + self.num_local_token_groups
        num_imgs = len(image_list) // len_per_img
        local_view_idx = list(
            itertools.chain(
                *[
                    list(range(i * len_per_img + 2, (i + 1) * len_per_img))
                    for i in range(num_imgs)
                ]
            )
        )
        for i, idx in enumerate(local_view_idx):
            view_idx = i % self.num_local_token_groups
            box_tensor = gt_instances[idx].gt_boxes.tensor
            box_hs = box_tensor[:, 3] - box_tensor[:, 1]
            local_box_hs = box_hs / self.num_local_token_groups
            box_tensor[:, 1] += view_idx * local_box_hs
            box_tensor[:, 3] = box_tensor[:, 1] + local_box_hs

            org_box_tensor = gt_instances[idx].org_gt_boxes.tensor
            org_box_hs = org_box_tensor[:, 3] - org_box_tensor[:, 1]
            local_org_box_hs = org_box_hs / self.num_local_token_groups
            org_box_tensor[:, 1] += view_idx * local_org_box_hs
            org_box_tensor[:, 3] = org_box_tensor[:, 1] + local_org_box_hs
        # NOTE 2. student output
        features = self.backbone(image_list.tensor)
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        feat_seq = features.flatten(2).transpose(1, 2)  # B x N x C

        out_seq_det, out_seq_reid = self.vit_student(
            feat_seq,
            image_list.tensor.shape,
            False,
            self.local_token_groups,
            self.local_token_pos_init,
            self.local_token_pos_mid,
            self.local_token_pos_mid_reid,
        )
        out_seq_det = out_seq_det[:, -self.num_queries :, :]
        del feat_seq
        reid_feats, out_feat = (
            out_seq_reid[:, -self.num_queries :, :],
            out_seq_reid[:, 1 : -self.num_queries, :].detach(),
        )
        del out_seq_reid
        reid_feats = self.bn_neck(reid_feats.transpose(1, 2)).transpose(1, 2)
        out_feat = out_feat.reshape(
            (-1, features.shape[-2], features.shape[-1], features.shape[1])
        ).permute(0, 3, 1, 2)
        del features
        # vis_featmap = feat_out.detach()
        outputs_class, outputs_coord = [], []
        out_seq_det = out_seq_det.reshape(
            (-1, len_per_img, out_seq_det.shape[-2], out_seq_det.shape[-1])
        )
        for vi in range(2):
            outputs_class.append(self.class_embed(out_seq_det[:, vi : vi + 1]))
            outputs_coord.append(self.bbox_embed(out_seq_det[:, vi : vi + 1]).sigmoid())
        for vi in range(2, len_per_img):
            outputs_class.append(
                self.class_embed_ext[vi - 2](out_seq_det[:, vi : vi + 1])
            )
            outputs_coord.append(
                self.bbox_embed_ext[vi - 2](out_seq_det[:, vi : vi + 1]).sigmoid()
            )
        del out_seq_det
        outputs_class = torch.cat(outputs_class, dim=1).flatten(0, 1)
        outputs_coord = torch.cat(outputs_coord, dim=1).flatten(0, 1)

        out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
        loss_dict = {}
        loss_det, matches = self.criterion(out, gt_instances)
        weight_dict = self.criterion.weight_dict
        for k in loss_det.keys():
            if k in weight_dict:
                loss_det[k] *= weight_dict[k]
        loss_dict.update(loss_det)
        if self.use_focal:
            det_scores = out["pred_logits"].detach().sigmoid()[..., 0]
        else:
            det_scores = out["pred_logits"].detach().softmax(-1)[..., 0]
        pred_box_xyxy = out["pred_boxes"].clone().detach()
        pred_box_xyxy[..., :2] -= pred_box_xyxy[..., 2:] / 2
        pred_box_xyxy[..., 2:] += pred_box_xyxy[..., :2]
        assign_ids = self.id_assigner(
            pred_box_xyxy,
            det_scores,
            [
                inst.gt_boxes.convert_mode(BoxMode.XYXY_REL, inst.image_size).tensor
                for inst in gt_instances
            ],
            [inst.gt_pids for inst in gt_instances],
            match_indices=matches["matches"],
        )
        assign_ids = (
            assign_ids.reshape((num_imgs, len_per_img, -1)).transpose(0, 1).flatten(1)
        )
        reid_feats = (
            reid_feats.reshape((num_imgs, len_per_img, reid_feats.shape[-2], -1))
            .transpose(0, 1)
            .flatten(1, 2)
        )
        pos_feats = []
        for vi in range(len_per_img):
            sorted_idxs_vi = torch.argsort(assign_ids[vi])
            sorted_feats = reid_feats[vi][sorted_idxs_vi]
            sorted_ids = assign_ids[vi][sorted_idxs_vi]
            pos_feats.append(sorted_feats[sorted_ids > -2])
        pos_feats = torch.cat(pos_feats, dim=0)
        student_out = self.head_student(pos_feats)

        # NOTE 3. teacher output
        # only global views
        global_view_idx = list(
            itertools.chain(
                *[[i * len_per_img, i * len_per_img + 1] for i in range(num_imgs)]
            )
        )
        image_list_g = image_list.select_by_indices(global_view_idx)
        assign_ids_g = torch.stack([assign_ids[i] for i in global_view_idx])
        features = self.backbone_teacher(image_list_g.tensor)
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        feat_seq = features.flatten(2).transpose(1, 2)  # B x N x C
        out_seq_reid = self.vit_teacher.forward_features_reid(
            feat_seq, image_list_g.tensor.shape
        )
        reid_feats = out_seq_reid[:, -self.num_queries :, :]
        del out_seq_reid
        reid_feats = self.bn_neck_teacher(reid_feats.transpose(1, 2)).transpose(1, 2)

        assign_ids_g = assign_ids[:2]
        reid_feats = (
            reid_feats.reshape((num_imgs, 2, reid_feats.shape[-2], -1))
            .transpose(0, 1)
            .flatten(1, 2)
        )
        pos_feats = []
        for vi in range(2):
            sorted_idxs_vi = torch.argsort(assign_ids_g[vi])
            sorted_feats = reid_feats[vi][sorted_idxs_vi]
            sorted_ids = assign_ids_g[vi][sorted_idxs_vi]
            pos_feats.append(sorted_feats[sorted_ids > -2])
        pos_feats = torch.cat(pos_feats, dim=0)
        teacher_out = self.head_teacher(pos_feats)

        # NOTE 4. dino loss
        loss_dict.update(
            {
                "loss_dino": self.dino_loss(
                    student_out, teacher_out, get_event_storage().iter
                )
            }
        )
        # NOTE 5. visualize
        # TODO fix the inconsistency between pred ids and vis
        with torch.no_grad():  # for vis only
            pred_instances_list = []
            featmap = out_feat  # vis_featmap
            for bi, (bi_boxes, bi_scores, bi_pids) in enumerate(
                zip(
                    pred_box_xyxy,
                    det_scores,
                    assign_ids,
                )
            ):
                pred_instances = Instances(
                    None,
                    pred_scores=bi_scores,
                    pred_boxes=Boxes(bi_boxes, BoxMode.XYXY_REL).convert_mode(
                        BoxMode.XYXY_ABS, gt_instances[bi].image_size
                    ),
                    assign_ids=bi_pids,
                )
                pred_instances_list.append(pred_instances)

        return pred_instances_list, featmap, loss_dict


# NOTE only use 1 global view
class VitDetWithExtTokens(VisionTransformer):
    def forward_features_with_tokens(
        self,
        feat_seq,
        imgt_shape,
        token_ext,
        init_pos_ext,
        mid_pos_ext=None,
    ):
        # import pdb;pdb.set_trace()
        B, H, W = imgt_shape[0], imgt_shape[2], imgt_shape[3]
        len_per_img = 1 + token_ext.shape[1]
        num_imgs = B // len_per_img
        all_det_tokens = torch.cat(
            [
                self.det_token.unsqueeze(1),
                token_ext,
            ],
            dim=1,
        )
        all_det_tokens = all_det_tokens.expand(num_imgs, -1, -1, -1).flatten(0, 1)
        all_det_token_init_pos = torch.cat(
            [
                self.pos_embed[:, -self.det_token_num :].unsqueeze(1),
                init_pos_ext,
            ],
            dim=1,
        )
        all_det_token_init_pos = all_det_token_init_pos.expand(
            num_imgs, -1, -1, -1
        ).flatten(0, 1)
        if self.has_mid_pe:
            all_det_token_mid_pos = torch.cat(
                [
                    self.mid_pos_embed[:, :, -self.det_token_num :].unsqueeze(2),
                    mid_pos_ext,
                ],
                dim=2,
            )
            all_det_token_mid_pos = all_det_token_mid_pos.expand(
                -1, num_imgs, -1, -1, -1
            ).flatten(1, 2)
        else:
            all_det_token_mid_pos = None

        x = feat_seq
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(
                self.pos_embed, img_size=(H, W)
            )
        else:
            temp_pos_embed = self.pos_embed
        patch_pos = temp_pos_embed[:, : -self.det_token_num].expand(B, -1, -1)
        temp_pos_embed = torch.cat([patch_pos, all_det_token_init_pos], dim=1)

        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed
            patch_pos = temp_mid_pos_embed[:, :, : -self.det_token_num].expand(
                -1, B, -1, -1
            )
            temp_mid_pos_embed = torch.cat([patch_pos, all_det_token_mid_pos], dim=2)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        det_token = all_det_tokens
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(len((self.blocks))):
            # x = self.blocks[i](x)
            x = checkpoint.checkpoint(self.blocks[i], x)  # saves mem, takes time
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]
        x = self.norm(x)

        return x

    def forward(self, feat_seq, imgt_shape, return_attention=False, *args, **kws):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(feat_seq, imgt_shape)
        else:
            if self.training:
                x = self.forward_features_with_tokens(
                    feat_seq, imgt_shape, *args, **kws
                )
            else:
                x = self.forward_features(feat_seq, imgt_shape)
            return x


import random


@META_ARCH_REGISTRY.register()
class VitPDWithLocal(SearchBase):
    @configurable
    def __init__(
        self,
        cfg,
        transformer,
        criterion,
        num_classes,
        num_queries,
        in_features,
        id_assigner,
        do_nms,
        num_local_tkgroups,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        # vitpara
        self.transformer = transformer
        hidden_dim = transformer.hidden_dim
        self.use_focal = criterion.use_focal
        if self.use_focal:
            self.class_embed = MLP(
                hidden_dim, hidden_dim, num_classes, 3
            )  # to avoid loading coco parameters
        else:
            self.class_embed = MLP(
                hidden_dim, hidden_dim, num_classes + 1, 3
            )  # to avoid loading coco parameters
        self.bbox_embed = MLP(
            hidden_dim, hidden_dim, 4, 3
        )  # to avoid loading coco parameters
        self.criterion = criterion
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.id_assigner = id_assigner
        self.in_features = in_features
        self.cfg = cfg
        self.do_nms = do_nms

        self.class_embed_ext = nn.ModuleList(
            [deepcopy(self.class_embed) for _ in range(num_local_tkgroups)]
        )
        self.bbox_embed_ext = nn.ModuleList(
            [deepcopy(self.bbox_embed) for _ in range(num_local_tkgroups)]
        )
        self.num_local_token_groups = num_local_tkgroups

        with torch.no_grad():
            # local tokens
            self.local_token_groups = nn.Parameter(
                torch.stack(
                    [
                        self.transformer.det_token.clone()
                        for _ in range(num_local_tkgroups)
                    ],
                    dim=1,
                )
            )
            # local token pos
            self.local_token_pos_init = nn.Parameter(
                torch.stack(
                    [
                        self.transformer.pos_embed[
                            :, -self.transformer.det_token_num :, :
                        ].clone()
                        for _ in range(num_local_tkgroups)
                    ],
                    dim=1,
                )
            )
            self.local_token_pos_mid = (
                nn.Parameter(
                    torch.stack(
                        [
                            self.transformer.mid_pos_embed[
                                :, :, -self.transformer.det_token_num :, :
                            ].clone()
                            for _ in range(num_local_tkgroups)
                        ],
                        dim=2,
                    )
                )
                if self.transformer.has_mid_pe
                else None
            )

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = build_id_assigner(cfg)
        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        dino_cfg = cfg.PERSON_SEARCH.DINO

        vit = VitDetWithExtTokens(
            pretrain_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size[0]
            if isinstance(patch_embed.patch_size, tuple)
            else patch_embed.patch_size,
            embed_dim=patch_embed.embed_dim,
            num_patches=patch_embed.num_patches,
            depth=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            is_distill=tr_cfg.DEIT,
        )
        det_cfg = cfg.PERSON_SEARCH.DET
        vit.finetune_det(
            img_size=tr_cfg.INIT_PE_SIZE,
            det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
            mid_pe_size=tr_cfg.MID_PE_SIZE,
        )
        ret["transformer"] = vit

        assert cfg.SOLVER.EPOCH_ITERS > 0, "Invalid SOLVER.EPOCH_ITERS !"

        ret["num_local_tkgroups"] = dino_cfg.NUM_LOCAL_TOKEN_GROUPS

        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES

        ret["do_nms"] = det_cfg.MODEL.DO_NMS
        return ret

    def forward(self, input_list):
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
            if self.training:
                input_list = list(
                    itertools.chain(
                        *[
                            [item["global"][random.randint(0, len(item["global"]) - 1)]]
                            + item["local"]
                            for item in input_list
                        ]
                    )
                )
                gt_instances = [gti["instances"].to(self.device) for gti in input_list]
                image_list = self.preprocess_input(input_list)
                preds, feat_maps, losses = self.forward_gallery(
                    image_list, gt_instances
                )
                self.visualize_training(image_list, gt_instances, preds, feat_maps)
                return losses
            else:
                image_list = self.preprocess_input(input_list)
                gt_instances = [gti["instances"].to(self.device) for gti in input_list]
                return self.forward_gallery(image_list, gt_instances)  # preds only

    def forward_query(self, image_list, gt_instances):
        return [Instances(gt_instances[i].image_size) for i in range(len(gt_instances))]

    def forward_gallery_eval(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        feat_seq = features.flatten(2).transpose(1, 2)  # B x N x C
        out_seq = self.transformer(feat_seq, image_list.tensor.shape)
        out_seq = out_seq[:, -self.num_queries :, :]
        # vis_featmap = feat_out.detach()
        outputs_class = self.class_embed(out_seq)
        outputs_coord = self.bbox_embed(out_seq).sigmoid()
        del out_seq
        pred_instances_list = []
        for bi, (bi_boxes, bi_logits) in enumerate(
            zip(
                outputs_coord,
                outputs_class,
            )
        ):
            if self.use_focal:
                pred_scores = bi_logits.sigmoid()
            else:
                pred_scores = bi_logits.softmax(-1)[..., 0]
            org_h, org_w = gt_instances[bi].org_img_size
            h, w = gt_instances[bi].image_size
            pred_boxes = Boxes(bi_boxes, BoxMode.CCWH_REL).convert_mode(
                BoxMode.XYXY_ABS, gt_instances[bi].image_size
            )
            pred_boxes.scale(org_w / w, org_h / h)
            if self.do_nms:
                keep = batched_nms(
                    pred_boxes.tensor,
                    pred_scores,
                    torch.zeros_like(pred_scores, dtype=torch.int64),
                    0.5,
                )
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

            pred_instances = Instances(
                None,
                pred_scores=pred_scores,
                pred_boxes=pred_boxes,
                reid_feats=pred_boxes.tensor,  # trivial impl for eval
            )
            pred_instances_list.append(pred_instances)

        return pred_instances_list

    def forward_gallery(self, image_list, gt_instances):
        if not self.training:
            return self.forward_gallery_eval(image_list, gt_instances)
        # NOTE modify local-view annotaions
        len_per_img = 1 + self.num_local_token_groups
        num_imgs = len(image_list) // len_per_img
        local_view_idx = list(
            itertools.chain(
                *[
                    list(range(i * len_per_img + 1, (i + 1) * len_per_img))
                    for i in range(num_imgs)
                ]
            )
        )
        for i, idx in enumerate(local_view_idx):
            view_idx = i % self.num_local_token_groups
            box_tensor = gt_instances[idx].gt_boxes.tensor
            box_hs = box_tensor[:, 3] - box_tensor[:, 1]
            local_box_hs = box_hs / self.num_local_token_groups
            box_tensor[:, 1] += view_idx * local_box_hs
            box_tensor[:, 3] = box_tensor[:, 1] + local_box_hs

            org_box_tensor = gt_instances[idx].org_gt_boxes.tensor
            org_box_hs = org_box_tensor[:, 3] - org_box_tensor[:, 1]
            local_org_box_hs = org_box_hs / self.num_local_token_groups
            org_box_tensor[:, 1] += view_idx * local_org_box_hs
            org_box_tensor[:, 3] = org_box_tensor[:, 1] + local_org_box_hs

        features = self.backbone(image_list.tensor)
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        feat_seq = features.flatten(2).transpose(1, 2)  # B x N x C

        out_seq_det = self.transformer(
            feat_seq,
            image_list.tensor.shape,
            False,
            self.local_token_groups,
            self.local_token_pos_init,
            self.local_token_pos_mid,
        )
        del feat_seq
        out_seq_det, out_feat = (
            out_seq_det[:, -self.num_queries :, :],
            out_seq_det[:, 1 : -self.num_queries, :].detach(),
        )
        out_feat = out_feat.reshape(
            (-1, features.shape[-2], features.shape[-1], features.shape[1])
        ).permute(0, 3, 1, 2)
        del features
        # vis_featmap = feat_out.detach()
        outputs_class, outputs_coord = [], []
        out_seq_det = out_seq_det.reshape(
            (-1, len_per_img, out_seq_det.shape[-2], out_seq_det.shape[-1])
        )
        for vi in range(1):
            outputs_class.append(self.class_embed(out_seq_det[:, vi : vi + 1]))
            outputs_coord.append(self.bbox_embed(out_seq_det[:, vi : vi + 1]).sigmoid())
        for vi in range(1, len_per_img):
            outputs_class.append(
                self.class_embed_ext[vi - 1](out_seq_det[:, vi : vi + 1])
            )
            outputs_coord.append(
                self.bbox_embed_ext[vi - 1](out_seq_det[:, vi : vi + 1]).sigmoid()
            )
        del out_seq_det
        outputs_class = torch.cat(outputs_class, dim=1).flatten(0, 1)
        outputs_coord = torch.cat(outputs_coord, dim=1).flatten(0, 1)

        out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
        loss_dict = {}
        loss_det, matches = self.criterion(out, gt_instances)
        weight_dict = self.criterion.weight_dict
        for k in loss_det.keys():
            if k in weight_dict:
                loss_det[k] *= weight_dict[k]
        loss_dict.update(loss_det)
        if self.use_focal:
            det_scores = out["pred_logits"].detach().sigmoid()[..., 0]
        else:
            det_scores = out["pred_logits"].detach().softmax(-1)[..., 0]
        pred_box_xyxy = out["pred_boxes"].clone().detach()
        pred_box_xyxy[..., :2] -= pred_box_xyxy[..., 2:] / 2
        pred_box_xyxy[..., 2:] += pred_box_xyxy[..., :2]
        assign_ids = self.id_assigner(
            pred_box_xyxy,
            det_scores,
            [
                inst.gt_boxes.convert_mode(BoxMode.XYXY_REL, inst.image_size).tensor
                for inst in gt_instances
            ],
            [inst.gt_pids for inst in gt_instances],
            match_indices=matches["matches"],
        )

        with torch.no_grad():  # for vis only
            pred_instances_list = []
            featmap = out_feat  # vis_featmap
            for bi, (bi_boxes, bi_scores, bi_pids) in enumerate(
                zip(
                    pred_box_xyxy,
                    det_scores,
                    assign_ids,
                )
            ):
                if self.do_nms:
                    keep = batched_nms(
                        bi_boxes,
                        bi_scores,
                        torch.zeros_like(bi_scores, dtype=torch.int64),
                        0.5,
                    )
                    bi_boxes = bi_boxes[keep]
                    bi_scores = bi_scores[keep]
                    bi_pids = bi_pids[keep]
                pred_instances = Instances(
                    None,
                    pred_scores=bi_scores,
                    pred_boxes=Boxes(bi_boxes, BoxMode.XYXY_REL).convert_mode(
                        BoxMode.XYXY_ABS, gt_instances[bi].image_size
                    ),
                    assign_ids=bi_pids,
                )
                pred_instances_list.append(pred_instances)

        return pred_instances_list, featmap, loss_dict


@META_ARCH_REGISTRY.register()
class VitPDWithLocalStochastic(VitPDWithLocal):
    @classmethod
    def from_config(cls, cfg):
        assert (
            cfg.SOLVER.IMS_PER_BATCH
            % (cfg.PERSON_SEARCH.DINO.NUM_LOCAL_TOKEN_GROUPS + 1)
            == 0
        ), "Invalid batch size !"
        ret = super().from_config(cfg)
        return ret

    def forward(self, input_list):
        return super(VitPDWithLocal, self).forward(input_list)
