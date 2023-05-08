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
from psd2.modeling.extend.swin import SwinTransformerWithSemanticCtrl,SwinTransformerWithAllSemanticCtrl
from torch.utils import checkpoint
from fvcore.nn import sigmoid_focal_loss_jit as sigmoid_focal_loss

@META_ARCH_REGISTRY.register()
class VidtPromptPDWithNeck(SearchBase):
    """NOTE vidt with neck"""

    @configurable
    def __init__(
        self,
        cfg,
        transformer,
        det_neck,
        box_refine,
        aux_loss,
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
        self.det_neck=det_neck
        self.aux_loss = aux_loss
        self.with_box_refine = box_refine
        hidden_dim = det_neck.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(
            hidden_dim, hidden_dim, 4, 3
        )  # to avoid loading coco parameters

        num_swin_outs = len(transformer.num_channels)
        input_proj_list = []
        for _ in range(num_swin_outs):
            in_channels = transformer.num_channels[_]
            input_proj_list.append(nn.Sequential(
                  # This is 1x1 conv -> so linear layer
                  nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                  nn.GroupNorm(32, hidden_dim),
                ))
        self.input_proj = nn.ModuleList(input_proj_list)

        # initialize the projection layer for [PATCH] tokens
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # channel dim reduction for [DET] tokens
        self.tgt_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(self.transformer.num_channels[-2], hidden_dim, kernel_size=1),
              nn.GroupNorm(32, hidden_dim),
            )
        # channel dim reductionfor [DET] learnable pos encodings
        self.query_pos_proj = nn.Sequential(
              # This is 1x1 conv -> so linear layer
              nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
              nn.GroupNorm(32, hidden_dim),
            )
        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # initialize projection layer for [DET] tokens and encodings
        nn.init.xavier_uniform_(self.tgt_proj[0].weight, gain=1)
        nn.init.constant_(self.tgt_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.query_pos_proj[0].weight, gain=1)
        nn.init.constant_(self.query_pos_proj[0].bias, 0)

        # the prediction is made for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = det_neck.decoder.num_layers + 1

        # set up all required nn.Module for additional techniques
        if box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.det_neck.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.det_neck.decoder.bbox_embed = None

        self.criterion = criterion
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.in_features = in_features
        self.cfg = cfg
        self.id_assigner = id_assigner

        self.freeze_params()

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training=mode
        if mode:
            # training:
            self.backbone.eval()
            self.transformer.eval()
            self.transformer.layers[-1].downsample.train()
            for n,m in self.transformer.named_modules():
                if "det" in n:
                    m.train()
            self.det_neck.train()
            self.input_proj.train()
            self.tgt_proj.train()
            self.query_pos_proj.train()
            self.class_embed.train()
            self.bbox_embed.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
                

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
            method="vidt",
            det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
            pos_dim=tr_cfg.DET_POS_DIM,
            cross_indices=tr_cfg.DET_CROSS_INDICES,
        )
        ret["transformer"] = swin
        neck=DeformableTransformer(
            d_model=det_cfg.MODEL.REDUCED_DIM,
            nhead=det_cfg.MODEL.NECK.NHEADS,
            num_decoder_layers=det_cfg.MODEL.NECK.NUM_DEC_LAYERS,
            dim_feedforward=det_cfg.MODEL.NECK.DIM_FEEDFORWARD,
            dropout=det_cfg.MODEL.NECK.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=det_cfg.MODEL.NECK.NUM_FEATURE_LEVELS,
            dec_n_points=det_cfg.MODEL.NECK.DEC_N_POINTS,
            token_label=False,
        )
        ret["det_neck"]=neck
        ret["box_refine"]=det_cfg.MODEL.BOX_REFINE
        ret["aux_loss"]=det_cfg.LOSS.DEEP_SUPERVISION
        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        return ret
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    def freeze_params(self):
        for n, p in self.transformer.named_parameters():
            if "det" not in n and "layers.3.downsample" not in n: # Vidt add another downsampling module
                p.requires_grad = False
        for  p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.transformer.eval()
        self.transformer.layers[-1].downsample.train()
    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        img_padding_mask = image_list.mask
        if isinstance(features, dict):
            features = features[list(features.keys())[-1]]
        features, det_tgt, det_pos = self.transformer(
            features, img_padding_mask, image_list.tensor.shape
        )
        # [DET] token and encoding projection to compact representation for the input to the Neck-free transformer
        det_tgt = self.tgt_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        det_pos = self.query_pos_proj(det_pos.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        # [PATCH] token projection
        shapes = []
        for l, src in enumerate(features):
            shapes.append(src.shape[-2:])

        srcs = []
        for l, src in enumerate(features):
                srcs.append(self.input_proj[l](src))
        del features

        masks = []
        for l, src in enumerate(srcs):
            # resize mask
            shapes.append(src.shape[-2:])
            _mask = F.interpolate(img_padding_mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            assert img_padding_mask is not None

        outputs_classes = []
        outputs_coords = []

        # return the output of the neck-free decoder
        hs, init_reference, inter_references, enc_token_class_unflat = \
          self.det_neck(srcs, masks, det_tgt, det_pos)
        del det_pos,det_tgt
        out_feat_vis = [feat.detach() for feat in srcs]
        del srcs

        # perform predictions via the detection head
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[lvl](hs[lvl])
            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # stack all predictions made from each decoding layers
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # final prediction is made the last decoding layer
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # aux loss is defined by using the rest predictions
        if self.aux_loss and self.det_neck.decoder.num_layers > 0:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            loss_dict = {}
            loss_det, matches = self.criterion(out, gt_instances)
            weight_dict = self.criterion.weight_dict
            for k in loss_det.keys():
                if k in weight_dict:
                    loss_det[k] *= weight_dict[k]
            loss_dict.update(loss_det)
            det_scores = out["pred_logits"].detach().sigmoid().squeeze(-1)
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
                    outputs_coord[-1],
                    outputs_class[-1],
                )
            ):
                pred_scores = bi_logits.sigmoid().squeeze(-1)
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

@META_ARCH_REGISTRY.register()
class VidtPromptPDWithNeckLastSemantic(VidtPromptPDWithNeck):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["cfg"] = cfg
        ret["id_assigner"] = build_id_assigner(cfg)
        ret["criterion"] = SetCriterion(cfg)
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        patch_embed = ret["backbone"]
        swin = SwinTransformerWithAllSemanticCtrl(
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
            method="vidt",
            det_token_num=det_cfg.MODEL.NUM_PROPOSALS,
            pos_dim=tr_cfg.DET_POS_DIM,
            cross_indices=tr_cfg.DET_CROSS_INDICES,
        )
        ret["transformer"] = swin
        neck=DeformableTransformer(
            d_model=det_cfg.MODEL.REDUCED_DIM,
            nhead=det_cfg.MODEL.NECK.NHEADS,
            num_decoder_layers=det_cfg.MODEL.NECK.NUM_DEC_LAYERS,
            dim_feedforward=det_cfg.MODEL.NECK.DIM_FEEDFORWARD,
            dropout=det_cfg.MODEL.NECK.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=det_cfg.MODEL.NECK.NUM_FEATURE_LEVELS,
            dec_n_points=det_cfg.MODEL.NECK.DEC_N_POINTS,
            token_label=False,
        )
        ret["det_neck"]=neck
        ret["box_refine"]=det_cfg.MODEL.BOX_REFINE
        ret["aux_loss"]=det_cfg.LOSS.DEEP_SUPERVISION
        ret["num_classes"] = det_cfg.NUM_CLASSES
        ret["num_queries"] = det_cfg.MODEL.NUM_PROPOSALS
        ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        return ret
    
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training=mode
        if mode:
            # training:
            self.backbone.eval()
            self.transformer.eval()
            self.transformer.last_downsample.train()
            for n,m in self.transformer.named_module():
                if "det" in n:
                    m.train()
            self.det_neck.train()
            self.input_proj.train()
            self.tgt_proj.train()
            self.query_pos_proj.train()
            self.class_embed.train()
            self.bbox_embed.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    def freeze_params(self):
        for n, p in self.transformer.named_parameters():
            if "det" not in n and "last_downsample" not in n: # Vidt add another downsampling module
                p.requires_grad = False
        for  p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.transformer.eval()
        self.transformer.last_downsample.train()

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
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,focal_alpha,
        focal_gamma,):
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
        self.focal_loss_alpha = focal_alpha
        self.focal_loss_gamma = focal_gamma
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
    @classmethod
    def from_config(cls, cfg):
        det_cfg = cfg.PERSON_SEARCH.DET
        ret = dict(
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
            for i in range(det_cfg.MODEL.NECK.NUM_DEC_LAYERS - 1+1):
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


# Neck for vidt, not compared with builtin deformable transformer yet
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from psd2.layers.ms_deform_attn import MSDeformAttn

from timm.models.layers import DropPath


class DeformableTransformer(nn.Module):
    """ A Deformable Transformer for the neck in a detector

    The transformer encoder is completely removed for ViDT
    Parameters:
        d_model: the channel dimension for attention [default=256]
        nhead: the number of heads [default=8]
        num_decoder_layers: the number of decoding layers [default=6]
        dim_feedforward: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        return_intermediate_dec: whether to return all the indermediate outputs [default=True]
        num_feature_levels: the number of scales for extracted features [default=4]
        dec_n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
        token_label: whether to use the token label loss for training [default=False]. This is an additional trick
            proposed in  https://openreview.net/forum?id=LhbD74dsZFL (ICLR'22) for further improvement
    """

    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=True, num_feature_levels=4, dec_n_points=4,
                 drop_path=0., token_label=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points,
                                                          drop_path=drop_path)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.token_label = token_label

        self.reference_points = nn.Linear(d_model, 2)

        if self.token_label:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            self.token_embed = nn.Linear(d_model, 91)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.token_embed.bias.data = torch.ones(91) * bias_value

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, tgt, query_pos):
        """ The forward step of the decoder

        Parameters:
            srcs: [Patch] tokens
            masks: input padding mask
            tgt: [DET] tokens
            query_pos: [DET] token pos encodings

        Returns:
            hs: calibrated [DET] tokens
            init_reference_out: init reference points
            inter_references_out: intermediate reference points for box refinement
            enc_token_class_unflat: info. for token labeling
        """

        # prepare input for the Transformer decoder
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = src_flatten
        bs, _, c = memory.shape
        tgt = tgt # [DET] tokens
        query_pos = query_pos.expand(bs, -1, -1) # [DET] token pos encodings

        # prepare input for token label
        if self.token_label:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_token_class_unflat = None
        if self.token_label:
            enc_token_class = self.token_embed(output_memory)
            enc_token_class_unflat = []
            for st, (h, w) in zip(level_start_index, spatial_shapes):
                enc_token_class_unflat.append(enc_token_class[:, st:st+h*w, :].view(bs, h, w, 91))

        # reference points for deformable attention
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points # query_pos -> reference point

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_pos, mask_flatten)

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out, enc_token_class_unflat


class DeformableTransformerDecoderLayer(nn.Module):
    """ A decoder layer.

    Parameters:
        d_model: the channel dimension for attention [default=256]
        d_ffn: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        n_levels: the number of scales for extracted features [default=4]
        n_heads: the number of heads [default=8]
        n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, drop_path=0.):
        super().__init__()

        # [DET x PATCH] deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # [DET x DET] self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn for multi-heaed
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):

        # [DET] self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Multi-scale deformable cross-attention in Eq. (1) in the ViDT paper
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)

        if self.drop_path is None:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # ffn
            tgt = self.forward_ffn(tgt)
        else:
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.drop_path(self.dropout4(tgt2))
            tgt = self.norm3(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    """ A Decoder consisting of multiple layers

    Parameters:
        decoder_layer: a deformable decoding layer
        num_layers: the number of layers
        return_intermediate: whether to return intermediate resutls
    """

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """ The forwared step of the Deformable Decoder

        Parameters:
            tgt: [DET] tokens
            reference_poitns: reference points for deformable attention
            src: the [PATCH] tokens fattened into a 1-d sequence
            src_spatial_shapes: the spatial shape of each multi-scale feature map
            src_level_start_index: the start index to refer different scale inputs
            src_valid_ratios: the ratio of multi-scale feature maps
            query_pos: the pos encoding for [DET] tokens
            src_padding_mask: the input padding mask

        Returns:
            output: [DET] tokens calibrated (i.e., object embeddings)
            reference_points: A reference points

            If return_intermediate = True, output & reference_points are returned from all decoding layers
        """

        output = tgt
        intermediate = []
        intermediate_reference_points = []

        # iterative bounding box refinement (handling the [DET] tokens produced from Swin with RAM)
        if self.bbox_embed is not None:
            tmp = self.bbox_embed[0](output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()
        #

        if self.return_intermediate:
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            # deformable operation
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid+1](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            #

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)




