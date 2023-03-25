import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from .base import SearchBase
from ..build import META_ARCH_REGISTRY
from psd2.utils.events import get_event_storage
from psd2.config.config import configurable
from psd2.modeling.roi_heads.roi_heads import (
    Res5ROIHeads,
    add_ground_truth_to_proposals,
)
from psd2.modeling.box_regression import Box2BoxTransform
from psd2.structures import Boxes, Instances, ImageList, pairwise_iou, BoxMode
from psd2.layers import ShapeSpec, batched_nms, cross_entropy
from psd2.utils.events import get_event_storage
from psd2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from psd2.modeling.proposal_generator import build_proposal_generator
from psd2.modeling.id_assign import build_id_assigner
from psd2.layers.mem_matching_losses import OIMLoss

logger = logging.getLogger(__name__)

# GeneralizedRCNN as reference
@META_ARCH_REGISTRY.register()
class OimBaseline(SearchBase):
    @configurable
    def __init__(
        self,
        proposal_generator,
        roi_heads,
        id_assigner,
        id_net,
        bn_neck,
        oim_loss,
        late_cls=False,
        late_cls_layer=None,
        *args,
        **kws,
    ) -> None:
        super().__init__(*args, **kws)
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.id_assigner = id_assigner
        self.id_net = id_net
        self.bn_neck = bn_neck
        self.oim_loss = oim_loss
        self.late_cls = late_cls
        self.late_cls_layer = late_cls_layer

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        bk_output_shape = ret["backbone"].output_shape()
        ret["proposal_generator"] = build_proposal_generator(cfg, bk_output_shape)
        ret["roi_heads"] = Res5ROIHeadsPs(cfg, bk_output_shape)
        ret["id_assigner"] = build_id_assigner(cfg)
        feat_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
        with_bn_neck = cfg.PERSON_SEARCH.REID.MODEL.BN_NECK
        id_net = nn.Linear(2048, feat_dim)
        init.normal_(id_net.weight, std=0.01)
        init.constant_(id_net.bias, 0)
        ret["id_net"] = id_net
        if with_bn_neck:
            bn_neck = nn.BatchNorm1d(feat_dim)
            init.normal_(bn_neck.weight, std=0.01)
            init.constant_(bn_neck.bias, 0)
        else:
            bn_neck = nn.Identity()
        ret["bn_neck"] = bn_neck
        ret["oim_loss"] = OIMLoss(cfg)
        late_cls = cfg.PERSON_SEARCH.DET.MODEL.LATE_CLS
        if late_cls:
            late_cls_layer = nn.Linear(feat_dim, 2)
            init.normal_(late_cls_layer.weight, std=0.01)
            init.constant_(late_cls_layer.bias, 0)
        else:
            late_cls_layer = None
        ret["late_cls"] = late_cls
        ret["late_cls_layer"] = late_cls_layer
        return ret

    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances
        )
        pred_instances, box_embs, losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances
        )
        box_embs = self.id_net(box_embs)
        box_embs = self.bn_neck(box_embs)
        if self.late_cls:
            gt_classes = [
                torch.ones(len(instances_i), dtype=torch.int64, device=box_embs.device)
                for instances_i in pred_instances
            ]
            for gt_classes_i, (src_idxs, tgt_idxs) in enumerate(
                gt_classes, pos_match_indices
            ):
                gt_classes_i[src_idxs] = 0
            pred_logits = self.late_cls_layer(box_embs)
            if self.training:
                losses.update(
                    {
                        "loss_cls": cross_entropy(
                            pred_logits, torch.cat(gt_classes), reduction="mean"
                        )
                        * self.roi_heads.box_predictor.loss_weight.get("loss_cls", 1.0)
                    }
                )
            with torch.no_grad():
                pred_scores = F.softmax(pred_logits, dim=-1)[:, 0]
                pred_scores = torch.split(
                    pred_scores, [len(instances_i) for instances_i in pred_instances]
                )
                for i, instances_i in enumerate(pred_instances):
                    instances_i.pred_scores = pred_scores[i]
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
            reid_loss = self.oim_loss(box_embs, assign_ids)
            losses.update(reid_loss)
            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
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
        features = self.backbone(image_list.tensor)
        box_embs = self.roi_heads.get_box_embs(features, gt_instances)
        box_embs = self.id_net(box_embs)
        box_embs = self.bn_neck(box_embs)
        box_embs = F.normalize(box_embs, dim=-1)
        box_embs = torch.split(box_embs, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=box_embs[i])
            for i in range(len(box_embs))
        ]


class Res5ROIHeadsPs(Res5ROIHeads):
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["box_predictor"] = (
            FastRCNNOutputLayersBoxOnly(
                cfg, ShapeSpec(channels=2048, height=1, width=1)
            )
            if cfg.PERSON_SEARCH.DET.MODEL.LATE_CLS
            else FastRCNNOutputLayersPs(
                cfg, ShapeSpec(channels=2048, height=1, width=1)
            )
        )

        return ret

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
        if self.proposal_append_gt:
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
                box_embs,
                losses,
                pos_match_indices,
            )
        else:
            pred_instances = self.box_predictor.inference_unms(predictions, proposals)
            return pred_instances, box_embs, {}, None

    def get_box_embs(
        self, features: Dict[str, torch.Tensor], targets: Optional[List[Instances]]
    ):
        target_boxes = [x.gt_boxes for x in targets]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], target_boxes
        )
        box_embs = box_features.mean(dim=[2, 3])
        return box_embs


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


class FastRCNNOutputLayersBoxOnly(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(
                weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
            ),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        return None, proposal_deltas

    def inference_unms(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        all_boxes = self.predict_boxes(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        results = []
        # borrowed from fast_rcnn_inference
        for boxes, image_shape in zip(all_boxes, image_shapes):
            valid_mask = torch.isfinite(boxes).all(dim=1)
            if not valid_mask.all():
                boxes = boxes[valid_mask]

            num_bbox_reg_classes = boxes.shape[1] // 4
            # Convert to Boxes to use the `clip` function ...
            boxes = Boxes(boxes.reshape(-1, 4))
            boxes.clip(image_shape)
            boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

            if num_bbox_reg_classes == 1:
                boxes = boxes[:, 0]

            result = Instances(image_shape)
            result.pred_boxes = Boxes(boxes)
            results.append(result)
        return results
