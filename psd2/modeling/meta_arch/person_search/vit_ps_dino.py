from psd2.modeling.meta_arch.person_search.vit_ps import (
    VitDetPsParallel,
    trunc_normal_,
    VitPSPara,
)
import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import SearchBase
from ..build import META_ARCH_REGISTRY
from psd2.config.config import configurable
from psd2.structures import Boxes, Instances, BoxMode
from psd2.modeling.id_assign import build_id_assigner
from psd2.layers.mem_matching_losses import OIMLoss
from psd2.layers import batched_nms
from psd2.modeling.matcher import DtHungarianMatcher as HungarianMatcher
from fvcore.nn import sigmoid_focal_loss_jit as sigmoid_focal_loss
from torchvision.ops import generalized_box_iou, box_convert
from psd2.utils import comm
import math
import numpy as np
from functools import partial

from torch.utils import checkpoint
from copy import deepcopy


class ViTDetPsWithTokenGroups(VitDetPsParallel):
    def finetune_det(
        self,
        img_size=[800, 1344],
        det_token_num=100,
        det_extgroup_num=2,
        mid_pe_size=None,
    ):
        super().finetune_det(img_size, det_token_num, mid_pe_size)
        self.det_extgroup_num = det_extgroup_num
        self.det_token_extgroups = nn.Parameter(
            torch.zeros(1, det_extgroup_num, det_token_num, self.embed_dim)
        ).to(self.pos_embed.device)
        self.det_token_extgroups = trunc_normal_(self.det_token_extgroups, std=0.02)
        det_pos_embed_extgroups = torch.zeros(
            1,
            det_extgroup_num,
            det_token_num,
            self.embed_dim,
            device=self.pos_embed.device,
        )
        self.det_pos_embed_extgroups = trunc_normal_(det_pos_embed_extgroups, std=0.02)
        if mid_pe_size is not None:
            self.mid_tk_pos_embed_extgroups = nn.Parameter(
                torch.zeros(
                    self.depth - 1,
                    1,
                    det_extgroup_num,
                    det_token_num,
                    self.embed_dim,
                    device=self.pos_embed.device,
                )
            )
            trunc_normal_(self.mid_tk_pos_embed_extgroups, std=0.02)
            self.mid_tk_pos_embed_reid = nn.Parameter(
                torch.zeros(
                    self.depth_reid - 1,
                    1,
                    det_extgroup_num,
                    det_token_num,
                    self.embed_dim,
                    device=self.pos_embed.device,
                )
            )
            trunc_normal_(self.mid_tk_pos_embed_reid, std=0.02)

    def forward_features_eval(self, feat_seq, imgt_shape):
        return super().forward_features(feat_seq, imgt_shape)

    def forward_features(self, feat_seq, imgt_shape):
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
        patch_pos = (
            temp_pos_embed[:, : -self.det_token_num]
            .unsqueeze(1)
            .expand(B, self.det_extgroup_num + 1, -1, -1)
        )
        token_pos = temp_pos_embed[:, -self.det_token_num :].unsqueeze(1)
        token_pos = torch.cat([token_pos, self.det_pos_embed_extgroups], dim=1).expand(
            B, -1, -1, -1
        )
        temp_pos_embed = torch.cat([patch_pos, token_pos], dim=2).flatten(0, 1)

        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed
            patch_pos = (
                temp_mid_pos_embed[:, :, : -self.det_token_num]
                .unsqueeze(2)
                .expand(-1, B, self.det_extgroup_num + 1, -1, -1)
            )
            token_pos = temp_mid_pos_embed[:, :, -self.det_token_num :].unsqueeze(2)
            token_pos = torch.cat(
                [token_pos, self.det_pos_embed_extgroups], dim=2
            ).expand(-1, B, -1, -1, -1)
            temp_mid_pos_embed = torch.cat([patch_pos, token_pos], dim=3).flatten(1, 2)

        cls_tokens = self.cls_token.unsqueeze(1).expand(
            B * self.det_tkgroup_num, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        det_token = (
            torch.cat([self.det_token.unsqueeze(1), self.det_token_extgroups], dim=1)
            .expand(B, -1, -1, -1)
            .flatten(0, 1)
        )  # batch_size becomes B x (num_ext_groups + 1)
        x = x.unsqueeze(1).expand(-1, self.det_tkgroup_num, -1, -1).flatten(0, 1)
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
            patch_pos = (
                temp_mid_pos_embed[:, :, : -self.det_token_num]
                .unsqueeze(2)
                .expand(-1, B, self.det_extgroup_num + 1, -1, -1)
            )
            token_pos = temp_mid_pos_embed[:, :, -self.det_token_num :].unsqueeze(2)
            token_pos = torch.cat(
                [token_pos, self.det_pos_embed_extgroups], dim=2
            ).expand(-1, B, -1, -1, -1)
            temp_mid_pos_embed = torch.cat([patch_pos, token_pos], dim=3).flatten(1, 2)

        for i in range(self.depth_reid):
            reid_x = checkpoint.checkpoint(
                self.blocks_reid[i], reid_x
            )  # saves mem, takes time
            if self.has_mid_pe_reid:
                if i < (self.depth_reid - 1):
                    reid_x = reid_x + temp_mid_pos_embed[i]
        reid_x = self.norm_reid(reid_x)

        return det_x.reshape((B, self.det_extgroup_num + 1, -1, -1)), reid_x.reshape(
            (B, self.det_extgroup_num + 1, -1, -1)
        )

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
        patch_pos = (
            temp_pos_embed[:, : -self.det_token_num]
            .unsqueeze(1)
            .expand(B, self.det_extgroup_num + 1, -1, -1)
        )
        token_pos = temp_pos_embed[:, -self.det_token_num :].unsqueeze(1)
        token_pos = torch.cat([token_pos, self.det_pos_embed_extgroups], dim=1).expand(
            B, -1, -1, -1
        )
        temp_pos_embed = torch.cat([patch_pos, token_pos], dim=2).flatten(0, 1)

        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed
            patch_pos = (
                temp_mid_pos_embed[:, :, : -self.det_token_num]
                .unsqueeze(2)
                .expand(-1, B, self.det_extgroup_num + 1, -1, -1)
            )
            token_pos = temp_mid_pos_embed[:, :, -self.det_token_num :].unsqueeze(2)
            token_pos = torch.cat(
                [token_pos, self.det_pos_embed_extgroups], dim=2
            ).expand(-1, B, -1, -1, -1)
            temp_mid_pos_embed = torch.cat([patch_pos, token_pos], dim=3).flatten(1, 2)

        cls_tokens = self.cls_token.unsqueeze(1).expand(
            B * self.det_tkgroup_num, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        det_token = (
            torch.cat([self.det_token.unsqueeze(1), self.det_token_extgroups], dim=1)
            .expand(B, -1, -1, -1)
            .flatten(0, 1)
        )  # batch_size becomes B x (num_ext_groups + 1)
        x = x.unsqueeze(1).expand(-1, self.det_tkgroup_num, -1, -1).flatten(0, 1)
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
            patch_pos = (
                temp_mid_pos_embed[:, :, : -self.det_token_num]
                .unsqueeze(2)
                .expand(-1, B, self.det_extgroup_num + 1, -1, -1)
            )
            token_pos = temp_mid_pos_embed[:, :, -self.det_token_num :].unsqueeze(2)
            token_pos = torch.cat(
                [token_pos, self.det_pos_embed_extgroups], dim=2
            ).expand(-1, B, -1, -1, -1)
            temp_mid_pos_embed = torch.cat([patch_pos, token_pos], dim=3).flatten(1, 2)

        for i in range(self.depth_reid):
            reid_x = checkpoint.checkpoint(
                self.blocks_reid[i], reid_x
            )  # saves mem, takes time
            if self.has_mid_pe_reid:
                if i < (self.depth_reid - 1):
                    reid_x = reid_x + temp_mid_pos_embed[i]
        reid_x = self.norm_reid(reid_x)

        return reid_x.reshape((B, self.det_extgroup_num + 1, -1, -1))


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
        warmup_teacher_temp_epochs,
        nepochs,
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
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
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
class VitPSParaDINO(VitPSPara):
    @configurable()
    def __init__(
        self,
        num_ext_tkgroups,
        transformer_teacher,
        head_student,
        head_teacher,
        dino_loss,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.class_embed_ext = nn.ModuleList(
            [deepcopy(self.class_embed) for _ in range(num_ext_tkgroups)]
        )
        self.bbox_embed_ext = nn.ModuleList(
            [deepcopy(self.bbox_embed) for _ in range(num_ext_tkgroups)]
        )
        self.vit_student = self.transformer
        self.vit_teacher = transformer_teacher
        self.head_student = head_student
        self.head_teacher = head_teacher
        self.dino_loss = dino_loss
        for p in self.vit_teacher.parameters():
            p.requires_grad = False
        for p in self.head_teacher.parameters():
            p.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = super(VitPSPara, cls).from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = build_id_assigner(cfg)

        ret["oim_loss"] = None
        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        reid_cfg = cfg.PERSON_SEARCH.REID
        vit = ViTDetPsWithTokenGroups(
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
            det_extgroup_num=det_cfg.MODEL.NUM_EXT_TOKEN_GROUPS,
            mid_pe_size=tr_cfg.MID_PE_SIZE,
        )
        ret["transformer"] = vit
        vit = ViTDetPsWithTokenGroups(
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
            det_extgroup_num=det_cfg.MODEL.NUM_EXT_TOKEN_GROUPS,
            mid_pe_size=tr_cfg.MID_PE_SIZE,
        )
        ret["transformer_teacher"] = vit

        det_cfg = cfg.PERSON_SEARCH.DET
        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        reid_cfg = cfg.PERSON_SEARCH.REID
        if reid_cfg.MODEL.BN_NECK:
            bn_neck = nn.BatchNorm1d(reid_cfg.MODEL.EMB_DIM, eps=1e-6)
            bn_neck.bias.requires_grad_(False)
            nn.init.constant_(bn_neck.weight, 1.0)
            nn.init.constant_(bn_neck.bias, 0.0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["do_nms"] = det_cfg.MODEL.DO_NMS
        ret["do_cws"] = reid_cfg.CWS
        return ret
