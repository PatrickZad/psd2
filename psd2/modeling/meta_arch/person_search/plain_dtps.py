import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch import Tensor
from .base import SearchBase
from ..build import META_ARCH_REGISTRY
from psd2.config.config import configurable
from psd2.structures import Boxes, Instances, BoxMode
from psd2.modeling.id_assign import build_id_assigner
from psd2.layers.mem_matching_losses import OIMLoss
from psd2.modeling.matcher import DtHungarianMatcher as HungarianMatcher
from fvcore.nn import sigmoid_focal_loss_jit as sigmoid_focal_loss
from torchvision.ops import generalized_box_iou, box_convert
from psd2.utils import comm
import copy
import math
from typing import Optional
from psd2.layers.pos_encoding import PositionEmbeddingSine
from psd2.modeling.extend.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)


@META_ARCH_REGISTRY.register()
class PlainDTPS(SearchBase):
    @configurable()
    def __init__(
        self,
        transformer,
        pos_enc,
        criterion,
        num_classes,
        num_queries,
        in_features,
        id_assigner,
        reid_head,
        oim_loss,
        reid_query_aux,
        reid_query_pos,
        aux_loss=True,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.num_queries = num_queries
        self.transformer = transformer
        self.criterion = criterion
        self.id_assigner = id_assigner
        self.reid_head = reid_head
        self.oim_loss = oim_loss
        self.pos_enc = pos_enc
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.in_features = in_features
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        backbone_outshapes = self.backbone.output_shape()
        self.input_proj = nn.Conv2d(
            backbone_outshapes[in_features[-1]].channels, hidden_dim, kernel_size=1
        )
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.reid_query_aux = reid_query_aux
        self.reid_query_pos = reid_query_pos

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["id_assigner"] = build_id_assigner(cfg)

        ret["oim_loss"] = OIMLoss(cfg)
        ret["criterion"] = SetCriterion(cfg)
        ret["pos_enc"] = PositionEmbeddingSine(
            cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER.D_MODEL // 2, normalize=True
        )
        ret["transformer"] = Transformer(cfg)
        det_cfg = cfg.PERSON_SEARCH.DET
        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        ret["reid_head"] = ReidHead(cfg)
        ret["reid_query_aux"] = cfg.PERSON_SEARCH.REID.MODEL.TRANSFORMER.QUERY_AUX
        ret["reid_query_pos"] = cfg.PERSON_SEARCH.REID.MODEL.TRANSFORMER.QUERY_POS
        ret["aux_loss"] = det_cfg.LOSS.DEEP_SUPERVISION
        return ret

    def forward_gallery(self, image_list, gt_instances):

        features = self.backbone(image_list.tensor)
        features = features[self.in_features[-1]]

        org_mask = image_list.mask
        mask = F.interpolate(org_mask[None].float(), size=features.shape[-2:]).to(
            torch.bool
        )[0]
        src = self.input_proj(features)
        pos = self.pos_enc(mask)
        query_pos = self.query_embed.weight
        input_info, hs = self.transformer(src, mask, query_pos, pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        query_pos = query_pos.unsqueeze(0).repeat(len(image_list), 1, 1)
        if self.training and self.reid_query_aux:
            reid_feats = self.reid_head(
                hs.flatten(0, 1),
                input_info[-1]
                .unsqueeze(0)
                .repeat(outputs_class.shape[0], 1, 1, 1)
                .flatten(0, 1),
                memory_key_padding_mask=input_info[1]
                .unsqueeze(0)
                .repeat(outputs_class.shape[0], 1, 1)
                .flatten(0, 1),
                memory_pos=input_info[2]
                .unsqueeze(0)
                .repeat(outputs_class.shape[0], 1, 1, 1)
                .flatten(0, 1),
                query_pos=query_pos.unsqueeze(0)
                .repeat(outputs_class.shape[0], 1, 1, 1)
                .flatten(0, 1)
                if self.reid_query_pos
                else None,
            )
        else:
            reid_feats = self.reid_head(
                hs[-1],
                input_info[-1],
                memory_key_padding_mask=input_info[1],
                memory_pos=input_info[2],
                query_pos=query_pos if self.reid_query_pos else None,
            )
        if self.training:
            loss_dict = {}
            loss_loc, matches = self.criterion(
                out, gt_instances
            )  # the gt_classes has been modified for query-matching
            weight_dict = self.criterion.weight_dict
            for k in loss_loc.keys():
                if k in weight_dict:
                    loss_loc[k] *= weight_dict[k]
            loss_dict.update(loss_loc)
            all_assign_ids = []
            if self.aux_loss:
                for i, (out_a, mat_a) in enumerate(
                    zip(out["aux_outputs"], matches["aux_matches"])
                ):
                    assign_ids = self.id_assigner(
                        out_a["pred_boxes"],
                        out_a["pred_logits"],
                        [inst.gt_boxes.tensor for inst in gt_instances],
                        [inst.gt_pids for inst in gt_instances],
                        match_indices=mat_a,
                    )
                    all_assign_ids.append(assign_ids)
            assign_ids = self.id_assigner(
                out["pred_boxes"],
                out["pred_logits"],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=matches["matches"],
            )
            all_assign_ids.append(assign_ids)
            if self.reid_query_aux:
                oim_loss = self.oim_loss(
                    reid_feats.reshape(-1, reid_feats.shape[-1]),
                    torch.cat(all_assign_ids, dim=0).reshape(-1),
                )
            else:
                oim_loss = self.oim_loss(
                    reid_feats.reshape(-1, reid_feats.shape[-1]),
                    all_assign_ids[-1].reshape(-1),
                )
            loss_dict.update(oim_loss)
            with torch.no_grad():  # for vis only
                pred_instances_list = []
                featmap_flatten = input_info[-1]  # memory b x l x c
                featmap = featmap_flatten.reshape(
                    featmap_flatten.shape[0], src.shape[2], src.shape[3], -1
                )
                featmap = featmap.permute(0, 3, 1, 2)
                for bi, (bi_boxes, bi_logits, bi_pids) in enumerate(
                    zip(
                        out["pred_boxes"],
                        out["pred_logits"],
                        all_assign_ids[-1],
                    )
                ):
                    top_indices = torch.topk(bi_logits.view(-1), k=100)[1]
                    pred_boxes = bi_boxes[top_indices]
                    pred_scores = bi_logits[top_indices].sigmoid().view(-1)
                    assign_ids = bi_pids[top_indices]
                    pred_instances = Instances(
                        None,
                        pred_scores=pred_scores,
                        pred_boxes=Boxes(pred_boxes, BoxMode.CCWH_REL).convert_mode(
                            BoxMode.XYXY_ABS, gt_instances[bi].image_size
                        ),
                        assign_ids=assign_ids,
                    )
                    pred_instances_list.append(pred_instances)

            return pred_instances_list, featmap, loss_dict

        else:
            pred_instances_list = []
            for bi, (bi_boxes, bi_logits, bi_reid_feats) in enumerate(
                zip(
                    out["pred_boxes"],
                    out["pred_logits"],
                    reid_feats,
                )
            ):
                top_indices = torch.topk(bi_logits.view(-1), k=100)[1]
                pred_boxes = bi_boxes[top_indices]
                pred_scores = bi_logits[top_indices].sigmoid().view(-1)
                reid_feats = bi_reid_feats[top_indices]
                org_h, org_w = gt_instances[bi].org_img_size
                h, w = gt_instances[bi].image_size
                pred_boxes = Boxes(pred_boxes, BoxMode.CCWH_REL).convert_mode(
                    BoxMode.XYXY_ABS, gt_instances[bi].image_size
                )
                pred_boxes.scale(org_w / w, org_h / h)
                pred_instances = Instances(
                    None,
                    pred_scores=pred_scores,
                    pred_boxes=pred_boxes,
                    reid_feats=reid_feats,
                )
                pred_instances_list.append(pred_instances)

            return pred_instances_list

    def forward_query(self, image_list, gt_instances):
        return [Instances(gt_instances[i].image_size) for i in range(len(gt_instances))]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class SetCriterion(nn.Module):
    """
    NOTE assume pred boxes to be CCWH_REL, gt boxes to be XYXY_ABS
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
             outputs: batched dict
             targets: gt instances
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
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt.gt_classes = torch.zeros_like(bt.gt_classes)
            match_indices["enc_matches"] = []
            indices = self.matcher(enc_outputs, bin_targets)
            match_indices["enc_matches"].append(indices)
            for loss in self.losses:
                l_dict = self.get_loss(
                    loss,
                    enc_outputs,
                    bin_targets,
                    indices,
                    num_boxes,
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, match_indices


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


class ReidHead(TransformerDecoder):
    @configurable()
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm,
        bn_neck,
        return_intermediate=False,
    ):
        super().__init__(
            decoder_layer,
            num_layers,
            norm,
            return_intermediate,
        )
        self.bn_neck = bn_neck

    @classmethod
    def from_config(cls, cfg):
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        dec_layer = TransformerDecoderLayer(
            dt_cfg.D_MODEL,
            dt_cfg.NHEAD,
            dt_cfg.DIM_FEEDFORWARD,
            dt_cfg.DROPOUT,
            dt_cfg.ACTIVATION,
            False,
        )
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        # TODO check if norm needed
        if cfg.PERSON_SEARCH.REID.MODEL.TRANSFORMER.WITH_NORM:
            norm = nn.LayerNorm(dt_cfg.D_MODEL)
        else:
            norm = nn.Identity()
        return dict(
            decoder_layer=dec_layer,
            num_layers=cfg.PERSON_SEARCH.REID.MODEL.TRANSFORMER.NUM_DECODER_LAYERS,
            norm=norm,
            return_intermediate=dt_cfg.RETURN_INTERMEDIATE_DEC,
            bn_neck=bn_neck,
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=memory_pos,
                query_pos=query_pos,
            )
        output=self.norm(output)
        output = self.bn_neck(output.permute(0, 2, 1)).permute(0, 2, 1)
        if not self.training:
            output = F.normalize(output, dim=-1)

        return output
