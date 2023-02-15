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
from functools import partial
from psd2.modeling.extend.vit import (
    VisionTransformer,
    Block,
    trunc_normal_,
    QMaskVisionTransformer,
    QMaskBlock,
)
from torch.utils import checkpoint


@META_ARCH_REGISTRY.register()
class VitPS(SearchBase):
    @configurable()
    def __init__(
        self,
        cfg,
        transformer,
        criterion,
        num_classes,
        num_queries,
        in_features,
        id_assigner,
        reid_head,
        oim_loss,
        do_nms,
        do_cws,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
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
        self.reid_head = reid_head
        self.oim_loss = oim_loss
        self.in_features = in_features
        self.cfg = cfg
        self.do_nms = do_nms
        self.do_cws = do_cws

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = build_id_assigner(cfg)

        ret["oim_loss"] = OIMLoss(cfg)
        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        vit = VisionTransformer(
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
        rhead = ReidHead(
            input_img_size=tr_cfg.INIT_PE_SIZE,
            num_patches=patch_embed.num_patches,
            patch_size=patch_embed.patch_size[0]
            if isinstance(patch_embed.patch_size, tuple)
            else patch_embed.patch_size,
            pretrain_img_size=patch_embed.pretrain_img_size,
            num_queies=det_cfg.MODEL.NUM_PROPOSALS,
            depth=reid_cfg.MODEL.TRANSFORMER.DEPTH,
            embed_dim=reid_cfg.MODEL.EMB_DIM,
            num_heads=tr_cfg.NHEAD,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            is_distill=reid_cfg.MODEL.TRANSFORMER.DEIT,
            bn_neck=bn_neck,
        )
        ret["reid_head"] = rhead
        ret["do_nms"] = det_cfg.MODEL.DO_NMS
        ret["do_cws"] = reid_cfg.CWS
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        feat_seq = features.flatten(2).transpose(1, 2)  # B x N x C
        out_seq = self.transformer(feat_seq, image_list.tensor.shape)
        reid_feats = self.reid_head(out_seq, image_list)
        out_seq, out_feat = (
            out_seq[:, -self.num_queries :, :],
            out_seq[
                :, self.transformer.cls_token.shape[1] : -self.num_queries, :
            ].detach(),
        )
        out_feat = out_feat.reshape(
            (-1, features.shape[-2], features.shape[-1], features.shape[1])
        ).permute(0, 3, 1, 2)
        del features
        # vis_featmap = feat_out.detach()
        outputs_class = self.class_embed(out_seq)
        outputs_coord = self.bbox_embed(out_seq).sigmoid()
        del out_seq

        if self.training:
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
            oim_loss = self.oim_loss(
                reid_feats.reshape(-1, reid_feats.shape[-1]),
                assign_ids.reshape(-1),
            )
            loss_dict.update(oim_loss)
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
        else:
            pred_instances_list = []
            for bi, (bi_boxes, bi_logits, bi_reid_feats) in enumerate(
                zip(
                    outputs_coord,
                    outputs_class,
                    reid_feats,
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
                    bi_reid_feats = bi_reid_feats[keep]
                if self.do_cws:
                    # TODO disable cws on query features
                    bi_reid_feats = bi_reid_feats * pred_scores.unsqueeze(-1)

                pred_instances = Instances(
                    None,
                    pred_scores=pred_scores,
                    pred_boxes=pred_boxes,
                    reid_feats=bi_reid_feats,
                )
                pred_instances_list.append(pred_instances)

            return pred_instances_list

    def forward_query(self, image_list, gt_instances):
        return [Instances(gt_instances[i].image_size) for i in range(len(gt_instances))]


@META_ARCH_REGISTRY.register()
class VitPD(VitPS):
    """
    #TODO fix the bug
    def load_state_dict(self, *args, **kws):
        if "bbox_embed.layers.0.weight" not in args[0]:
            # pre-train
            outputs = super().load_state_dict(*args, **kws)
            det_cfg = self.cfg.PERSON_SEARCH.DET
            tr_cfg = self.cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
            self.transformer.finetune_det(
                img_size=tr_cfg.INIT_PE_SIZE,
                det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
                mid_pe_size=tr_cfg.MID_PE_SIZE,
            )
        else:
            # resume
            det_cfg = self.cfg.PERSON_SEARCH.DET
            tr_cfg = self.cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
            self.transformer.finetune_det(
                img_size=tr_cfg.INIT_PE_SIZE,
                det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
                mid_pe_size=tr_cfg.MID_PE_SIZE,
            )
            outputs = super().load_state_dict(*args, **kws)
        return outputs
    """

    @classmethod
    def from_config(cls, cfg):
        ret = super(VitPS, cls).from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = None

        ret["oim_loss"] = None
        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        vit = VisionTransformer(
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

        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        ret["reid_head"] = None
        ret["do_nms"] = det_cfg.MODEL.DO_NMS
        return ret

    def forward_gallery(self, image_list, gt_instances):
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

        if self.training:
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

            with torch.no_grad():  # for vis only
                pred_instances_list = []
                featmap = None  # vis_featmap
                for bi, (bi_boxes, bi_scores) in enumerate(
                    zip(
                        pred_box_xyxy,
                        det_scores,
                    )
                ):
                    pred_instances = Instances(
                        None,
                        pred_scores=bi_scores,
                        pred_boxes=Boxes(bi_boxes, BoxMode.XYXY_REL).convert_mode(
                            BoxMode.XYXY_ABS, gt_instances[bi].image_size
                        ),
                        assign_ids=bi_scores.new_zeros(
                            bi_scores.shape, dtype=torch.int32
                        ),
                    )
                    pred_instances_list.append(pred_instances)

            return pred_instances_list, featmap, loss_dict
        else:
            pred_instances_list = []
            for bi, (bi_boxes, bi_logits) in enumerate(
                zip(outputs_coord, outputs_class)
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

    def forward_query(self, image_list, gt_instances):
        raise NotImplementedError


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    @configurable()
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        use_focal,
        focal_alpha,
        focal_gamma,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = focal_alpha
            self.focal_loss_gamma = focal_gamma
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer("empty_weight", empty_weight)

    @classmethod
    def from_config(cls, cfg):
        det_cfg = cfg.PERSON_SEARCH.DET
        ret = dict(
            use_focal=det_cfg.LOSS.FOCAL.ENABLE,
            focal_alpha=det_cfg.LOSS.FOCAL.ALPHA,
            focal_gamma=det_cfg.LOSS.FOCAL.GAMMA,
            eos_coef=det_cfg.LOSS.WEIGHTS.NO_OBJECT,
            matcher=HungarianMatcher(cfg),
            num_classes=det_cfg.NUM_CLASSES,
        )
        weight_dict = {
            "loss_ce": det_cfg.LOSS.WEIGHTS.CLS,
            "loss_bbox": det_cfg.LOSS.WEIGHTS.L1,
            "loss_giou": det_cfg.LOSS.WEIGHTS.GIOU,
        }

        if det_cfg.LOSS.DEEP_SUPERVISION:
            aux_weight_dict = {}
            for i in range(det_cfg.MODEL.TRANSFORMER.NUM_DECODER_LAYERS - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        ret["weight_dict"] = weight_dict
        ret["losses"] = losses
        return ret

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t.gt_classes[J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        if self.use_focal:
            src_logits = src_logits.flatten(0, 1)
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[
                0
            ]
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1
            # comp focal loss.
            class_loss = (
                sigmoid_focal_loss(
                    src_logits,
                    labels,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                )
                / num_boxes
            )
            losses = {"loss_ce": class_loss}
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
            losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        src_boxes = box_convert(src_boxes, "cxcywh", "xyxy")
        xyxy_rel_target_boxes = [
            t.gt_boxes.convert_mode(BoxMode.XYXY_REL, t.image_size) for t in targets
        ]
        target_boxes = torch.cat(
            [t.tensor[i] for t, (_, i) in zip(xyxy_rel_target_boxes, indices)], dim=0
        )

        losses = {}
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        match_indices = {}

        # Retrieve the matching between the outputs of the last layer and the targets

        indices = self.matcher(outputs_without_aux, targets)
        match_indices["matches"] = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t.gt_boxes) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if comm.get_world_size() > 1:
            comm.synchronize()
            all_num_boxes = comm.all_gather(num_boxes)
            num_boxes = sum([num.to("cpu") for num in all_num_boxes])
            num_boxes = torch.clamp(num_boxes / comm.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            match_indices["aux_matches"] = []
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                match_indices["aux_matches"].append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_boxes,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, match_indices


class ReidHead(VisionTransformer):
    def __init__(
        self,
        input_img_size,
        num_patches,
        patch_size=16,
        pretrain_img_size=224,
        num_queies=100,
        depth=1,
        embed_dim=384,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        is_distill=False,
        bn_neck=nn.Identity(),
    ):
        super(VisionTransformer, self).__init__()

        self.img_size = pretrain_img_size
        self.depth = depth
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = num_patches

        if is_distill:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 2, self.embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.embed_dim)
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)

        self.bn_neck = bn_neck

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        # set finetune flag
        self.has_mid_pe = True if depth > 1 else False

        # finetune_det
        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

        self.det_token_num = num_queies

        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = torch.zeros(1, num_queies, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=0.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = (
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = input_img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = torch.nn.Parameter(
            torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        )
        self.img_size = input_img_size
        if self.has_mid_pe:
            self.mid_pos_embed = nn.Parameter(
                torch.zeros(
                    self.depth - 1,
                    1,
                    1
                    + (input_img_size[0] * input_img_size[1] // self.patch_size ** 2)
                    + num_queies,
                    self.embed_dim,
                )
            )
            trunc_normal_(self.mid_pos_embed, std=0.02)
            self.mid_pe_size = input_img_size

    def forward(self, x, image_list):
        # x:B x N x C
        t = image_list.tensor
        B, H, W = t.shape[0], t.shape[2], t.shape[3]

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

        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(len((self.blocks))):
            x = self.blocks[i](x)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        x = x[:, -self.det_token_num :, :]
        out = self.bn_neck(x.transpose(1, 2)).transpose(1, 2)
        if not self.training:
            out = F.normalize(out, dim=-1)
        return out


@META_ARCH_REGISTRY.register()
class QMaskVitPS(VitPS):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        vit = QMaskVisionTransformer(
            patch_embed=ret["backbone"],
            depth=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            norm_layer=nn.LayerNorm,
            is_distill=tr_cfg.DEIT,
        )
        det_cfg = cfg.PERSON_SEARCH.DET
        vit.finetune_det(
            det_token_start=tr_cfg.DET_TOKEN_START,
            shared_qsa=tr_cfg.SHARED_QSA,
            img_size=tr_cfg.INIT_PE_SIZE,
            det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
            mid_pe_size=tr_cfg.MID_PE_SIZE,
        )
        ret["transformer"] = vit

        det_cfg = cfg.PERSON_SEARCH.DET
        reid_cfg = cfg.PERSON_SEARCH.REID
        if reid_cfg.MODEL.BN_NECK:
            bn_neck = nn.BatchNorm1d(reid_cfg.MODEL.EMB_DIM)
            bn_neck.bias.requires_grad_(False)
            nn.init.constant_(bn_neck.weight, 1.0)
            nn.init.constant_(bn_neck.bias, 0.0)
        else:
            bn_neck = nn.Identity()
        rhead = QMaskReidHead(
            input_img_size=tr_cfg.INIT_PE_SIZE,
            num_patches=vit.num_patches,
            patch_size=vit.patch_size,
            pretrain_img_size=vit.patch_embed.pretrain_img_size,
            num_queies=det_cfg.MODEL.NUM_PROPOSALS,
            depth=reid_cfg.MODEL.TRANSFORMER.DEPTH,
            embed_dim=reid_cfg.MODEL.EMB_DIM,
            num_heads=tr_cfg.NHEAD,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            norm_layer=nn.LayerNorm,
            bn_neck=bn_neck,
            shared_qsa=tr_cfg.SHARED_QSA,
        )
        ret["reid_head"] = rhead
        return ret

    def load_state_dict(self, state_dict, strict: bool = True):
        attn_block_params = {}
        is_resume = False
        for k, v in state_dict.items():
            if "transformer.blocks" in k:
                attn_block_params[k] = v
                if "qsa" in k:
                    is_resume = True
                    break
        if not is_resume:
            for k, v in attn_block_params.items():
                kws = k.split(".")
                new_k = ".".join(kws[:3] + ["qsa"] + kws[3:])
                state_dict[new_k] = v
        # if self.transformer.shared_qsa and ""
        ret = super().load_state_dict(state_dict, strict)
        # TODO
        # init qsa with pretrained param
        return ret


import copy


class QMaskReidHead(QMaskVisionTransformer):
    def __init__(
        self,
        input_img_size,
        num_patches,
        patch_size=16,
        pretrain_img_size=224,
        num_queies=100,
        depth=1,
        embed_dim=384,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        bn_neck=nn.Identity(),
        shared_qsa=True,
    ):
        super(VisionTransformer, self).__init__()

        self.img_size = pretrain_img_size
        self.depth = depth
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                QMaskBlock(
                    len_q=num_queies,
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(self.embed_dim)

        self.bn_neck = bn_neck

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        # set finetune flag
        self.has_mid_pe = True if depth > 1 else False

        # finetune_det
        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

        self.det_token_num = num_queies

        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = torch.zeros(1, num_queies, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=0.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = (
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = input_img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = torch.nn.Parameter(
            torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        )
        self.img_size = input_img_size
        if self.has_mid_pe:
            self.mid_pos_embed = nn.Parameter(
                torch.zeros(
                    self.depth - 1,
                    1,
                    1
                    + (input_img_size[0] * input_img_size[1] // self.patch_size ** 2)
                    + num_queies,
                    self.embed_dim,
                )
            )
            trunc_normal_(self.mid_pos_embed, std=0.02)
            self.mid_pe_size = input_img_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 1)]
        qsa = Block(
            dim=self.embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
        )
        for i in range(self.depth):
            blk = self.blocks[i]
            if shared_qsa:
                blk.qsa = qsa
            else:
                blk.qsa = copy.deepcopy(qsa)

        self.shared_qsa = shared_qsa

    def forward(self, x, image_list):
        # x:B x N x C
        t = image_list.tensor
        B, H, W = t.shape[0], t.shape[2], t.shape[3]

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

        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(len((self.blocks))):
            x = self.blocks[i](x)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        x = x[:, -self.det_token_num :, :]
        out = self.bn_neck(x.transpose(1, 2)).transpose(1, 2)
        if not self.training:
            out = F.normalize(out, dim=-1)
        return out


class VitDetPsParallel(VisionTransformer):
    def __init__(self, depth_reid, *args, **kws):
        super().__init__(*args, **kws)
        self.norm_reid = copy.deepcopy(self.norm)
        self.blocks_reid = nn.ModuleList(
            [copy.deepcopy(self.blocks[i]) for i in range(-depth_reid, 0)]
        )
        self.para_start_idx = len(self.blocks) - depth_reid
        self.depth_reid = depth_reid

    def finetune_det(
        self,
        img_size=[800, 1344],
        det_token_num=100,
        mid_pe_size=None,
    ):
        super().finetune_det(img_size, det_token_num, mid_pe_size)
        if mid_pe_size == None:
            self.has_mid_pe_reid = False
            print("No mid pe reid")
        else:
            print("Has mid pe reid")
            self.mid_pos_embed_reid = nn.Parameter(
                torch.zeros(
                    self.depth_reid - 1,
                    1,
                    1
                    + (mid_pe_size[0] * mid_pe_size[1] // self.patch_size ** 2)
                    + det_token_num,
                    self.embed_dim,
                    device=self.pos_embed.device,
                )
            )
            trunc_normal_(self.mid_pos_embed_reid, std=0.02)
            self.has_mid_pe_reid = True

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

        for i in range(self.depth_reid):
            reid_x = checkpoint.checkpoint(
                self.blocks_reid[i], reid_x
            )  # saves mem, takes time
            if self.has_mid_pe_reid:
                if i < (self.depth_reid - 1):
                    reid_x = reid_x + temp_mid_pos_embed[i]
        reid_x = self.norm_reid(reid_x)

        return det_x, reid_x

    def forward_return_all_selfattention(self, feat_seq, imgt_shape):
        raise NotImplementedError


@META_ARCH_REGISTRY.register()
class VitPSPara(SearchBase):
    @configurable()
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
        oim_loss,
        do_nms,
        do_cws,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
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
        self.oim_loss = oim_loss
        self.in_features = in_features
        self.cfg = cfg
        self.do_nms = do_nms
        self.do_cws = do_cws

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = build_id_assigner(cfg)

        ret["oim_loss"] = OIMLoss(cfg)
        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        reid_cfg = cfg.PERSON_SEARCH.REID
        vit = VitDetPsParallel(
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

    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        feat_seq = features.flatten(2).transpose(1, 2)  # B x N x C
        out_seq_det, out_seq_reid = self.transformer(feat_seq, image_list.tensor.shape)
        out_seq_det = out_seq_det[:, -self.num_queries :, :]
        del feat_seq
        reid_feats, out_feat = (
            out_seq_reid[:, -self.num_queries :, :],
            out_seq_reid[
                :, self.transformer.cls_token.shape[1] : -self.num_queries, :
            ].detach(),
        )
        del out_seq_reid
        reid_feats = self.bn_neck(reid_feats.transpose(1, 2)).transpose(1, 2)
        if not self.training:
            reid_feats = F.normalize(reid_feats, dim=-1)
        out_feat = out_feat.reshape(
            (-1, features.shape[-2], features.shape[-1], features.shape[1])
        ).permute(0, 3, 1, 2)
        del features
        # vis_featmap = feat_out.detach()
        outputs_class = self.class_embed(out_seq_det)
        outputs_coord = self.bbox_embed(out_seq_det).sigmoid()
        del out_seq_det

        if self.training:
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
            oim_loss = self.oim_loss(
                reid_feats.reshape(-1, reid_feats.shape[-1]),
                assign_ids.reshape(-1),
            )
            loss_dict.update(oim_loss)
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
        else:
            pred_instances_list = []
            for bi, (bi_boxes, bi_logits, bi_reid_feats) in enumerate(
                zip(
                    outputs_coord,
                    outputs_class,
                    reid_feats,
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
                    bi_reid_feats = bi_reid_feats[keep]
                if self.do_cws:
                    # TODO disable cws on query features
                    bi_reid_feats = bi_reid_feats * pred_scores.unsqueeze(-1)

                pred_instances = Instances(
                    None,
                    pred_scores=pred_scores,
                    pred_boxes=pred_boxes,
                    reid_feats=bi_reid_feats,
                )
                pred_instances_list.append(pred_instances)

            return pred_instances_list

    def forward_query(self, image_list, gt_instances):
        return [Instances(gt_instances[i].image_size) for i in range(len(gt_instances))]
