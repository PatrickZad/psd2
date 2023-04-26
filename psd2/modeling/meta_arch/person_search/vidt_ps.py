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
from torchvision.ops import generalized_box_iou, box_convert
from psd2.utils import comm
import math
from functools import partial
from psd2.modeling.extend.swin import SwinTransformerWithSemanticCtrl
from torch.utils import checkpoint


@META_ARCH_REGISTRY.register()
class VidtPromptPD(SearchBase):
    """NOTE vidt wo neck"""

    @configurable
    def __init__(
        self,
        cfg,
        transformer,
        reduced_dim,
        criterion,
        num_classes,
        num_queries,
        in_features,
        id_assigner,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.transformer = transformer
        hidden_dim = transformer.num_channels[-1]
        self.input_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, reduced_dim, kernel_size=1),
            nn.GroupNorm(32, reduced_dim),
        )
        self.class_embed = MLP(
            reduced_dim, reduced_dim, num_classes + 1, 3
        )  # to avoid loading coco parameters
        self.bbox_embed = MLP(
            reduced_dim, reduced_dim, 4, 3
        )  # to avoid loading coco parameters
        self.criterion = criterion
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.in_features = in_features
        self.cfg = cfg
        self.id_assigner = id_assigner

        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)

        for n, p in self.transformer.named_parameters():
            if "det" not in n:
                p.requires_grad = False
        for  p in self.backbone.parameters():
            p.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = build_id_assigner(cfg)
        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        swin = SwinTransformerWithSemanticCtrl(
            semantic_weight=tr_cfg.SEMANTIC_WEIGHT,
            pretrain_img_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size[0]
            if isinstance(patch_embed.patch_size, tuple)
            else patch_embed.patch_size,
            embed_dim=patch_embed.embed_dim,
            depths=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            window_size=tr_cfg.WIN_SIZE,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
            norm_layer=nn.LayerNorm,
        )
        det_cfg = cfg.PERSON_SEARCH.DET
        swin.finetune_det(
            method="vidt_wo_neck",
            det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
            pos_dim=tr_cfg.DET_POS_DIM,
            cross_indices=tr_cfg.DET_CROSS_INDICES,
        )
        ret["transformer"] = swin

        det_cfg = cfg.PERSON_SEARCH.DET
        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        ret["reduced_dim"]=det_cfg.MODEL.REDUCED_DIM
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        img_padding_mask = image_list.mask
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        patch_outs, det_tgt, det_pos = self.transformer(
            features, img_padding_mask, image_list.tensor.shape
        )
        del det_pos
        out_feat_vis = [feat.detach() for feat in patch_outs]
        del patch_outs
        det_tgt = self.input_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        outputs_class = self.class_embed(det_tgt)
        outputs_coord = self.bbox_embed(det_tgt).sigmoid()
        del det_tgt

        if self.training:
            out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
            loss_dict = {}
            loss_det, matches = self.criterion(out, gt_instances)
            weight_dict = self.criterion.weight_dict
            for k in loss_det.keys():
                if k in weight_dict:
                    loss_det[k] *= weight_dict[k]
            loss_dict.update(loss_det)
            det_scores = out["pred_logits"].detach().softmax(-1)[..., 0]
            pred_box_xyxy = out["pred_boxes"].clone().detach()  # ccwh
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
                featmap = out_feat_vis  # vis_featmap
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
            for bi, (bi_boxes, bi_logits) in enumerate(
                zip(
                    outputs_coord,
                    outputs_class,
                )
            ):
                pred_scores = bi_logits.softmax(-1)[..., 0]
                org_h, org_w = gt_instances[bi].org_img_size
                h, w = gt_instances[bi].image_size
                pred_boxes = Boxes(bi_boxes, BoxMode.CCWH_REL).convert_mode(
                    BoxMode.XYXY_ABS, gt_instances[bi].image_size
                )
                pred_boxes.scale(org_w / w, org_h / h)

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


# NOTE simplified version of deformable detr criterion
class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    @configurable
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
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
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
    @classmethod
    def from_config(cls, cfg):
        det_cfg = cfg.PERSON_SEARCH.DET
        ret = dict(
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
        NOTE not focal loss
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

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]  # ccwh
        target_boxes = [
            t.gt_boxes.convert_mode(BoxMode.CCWH_REL, t.image_size) for t in targets
        ]
        target_boxes = torch.cat(
            [t.tensor[i] for t, (_, i) in zip(target_boxes, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_convert(src_boxes, "cxcywh", "xyxy"),
                box_convert(target_boxes, "cxcywh", "xyxy"),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
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
                        loss, aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, match_indices




