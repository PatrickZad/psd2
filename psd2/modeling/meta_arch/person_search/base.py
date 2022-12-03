import numpy as np
import torch
import torch.nn as nn
from ...backbone import build_backbone
from ..build import META_ARCH_REGISTRY
from psd2.structures import ImageList, BoxMode
from psd2.utils.events import get_event_storage
import torchvision.transforms.functional as tvF
from psd2.utils.visualizer import Visualizer, mlvl_pca_feat
from psd2.config.config import configurable

COLORS = ["r", "g", "b", "y", "c", "m"]
T_COLORS_BG = {
    "r": "white",
    "g": "white",
    "b": "white",
    "y": "black",
    "c": "black",
    "m": "white",
}


@META_ARCH_REGISTRY.register()
class SearchBase(nn.Module):
    @configurable()
    def __init__(self, backbone, pix_mean, pix_std, vis_period):
        super().__init__()
        self.backbone = backbone

        self.register_buffer("pixel_mean", torch.Tensor(pix_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pix_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.vis_period = vis_period

    @property
    def device(self):
        return self.pixel_mean.device

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
        }

    def preprocess_input(self, input_list):
        images = [x["image"].to(self.device) for x in input_list]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, input_list):
        """
        gt_instances attrs:
            image_size: hw (int, int)
            file_name: full path string
            image_id: filename string
            gt_boxes: Boxes
            gt_classes: tensor full with 0s
            gt_pids: person identity tensor in [-1, max_id]
            org_img_size: hw before augmentation (int, int)
            org_gt_boxes: Boxes before augmentation
        pred_instances attrs:
            pred_boxes: Boxes in augmentation range (resizing/cropping/flipping/...)
            pred_scores: tensor
            assign_ids: assigned person identities (during training only)
            reid_feats: tensor
        """
        if "query" in input_list[0].keys():
            q_img_list = self.preprocess_input([qi["query"] for qi in input_list])
            q_gt_instances = gt_instances = [
                x["query"]["instances"].to(self.device) for x in input_list
            ]
            q_out_instances = self.forward_query(q_img_list, q_gt_instances)
            return q_out_instances
        else:
            image_list = self.preprocess_input(input_list)
            if "instances" in input_list[0]:
                gt_instances = [x["instances"].to(self.device) for x in input_list]
            else:
                gt_instances = None
            if self.training:
                pred_instances, feat_maps, losses = self.forward_gallery(
                    image_list, gt_instances
                )
                self.visualize_training(
                    image_list, gt_instances, pred_instances, feat_maps
                )
                return losses
            else:
                return self.forward_gallery(
                    image_list, gt_instances
                )  # pred_instances only

    def forward_gallery(self, image_list, gt_instances):
        """
        Return cls_scores, bboxes and reid features
        """
        raise NotImplementedError

    def forward_query(self, image_list, gt_instances):
        """
        Return cls_scores, bboxes and reid features
        """
        raise NotImplementedError

    def visualize_training(
        self, image_list, gt_instances, pred_instances, feat_map=None
    ):
        if self.vis_period < 1:
            return
        storage = get_event_storage()
        if storage.iter % self.vis_period != 0:
            return

        def score_split(score, threds):
            for i, v in enumerate(threds[:-1]):
                if score >= threds[i] and score < threds[i + 1]:
                    return i

        threds = [0, 0.2, 0.5, 1]
        bs = len(gt_instances)
        if feat_map is not None:
            level_pcas = []
            if isinstance(featmap, torch.Tensor):
                featmap = featmap[None]
            if isinstance(featmap, dict):
                featmap = list(featmap.values())
            level_pcas = []
            for fmpi in featmap:
                level_pcas.append(mlvl_pca_feat(fmpi[None])[0])
            max_h, w = 0, 0
            level_pca_feats = []
            for bi in range(bs):
                for feat in level_pcas:
                    bi_pca = feat[bi]
                    level_pca_feats.append(bi_pca)
                    if bi_pca.shape[-2] > max_h:
                        max_h = bi_pca.shape[-2]
                    w += bi_pca.shape[-1]
                bg = torch.zeros(3, max_h, w, dtype=torch.float32)
                next_w = 0
                for feat in level_pcas:
                    bg[
                        :, : feat[bi].shape[-2], next_w : next_w + feat[bi].shape[-1]
                    ] = feat[bi]
                    next_w += feat[bi].shape[-1]
                storage.put_image("img_{}/feat".format(bi), bg / 255)
        for bi in range(bs):
            img_t = image_list.tensor[bi].cpu()
            img_rgb = (img_t.permute(1, 2, 0) * 255).numpy()
            visualize_gt = Visualizer(img_rgb.copy())
            boxes = gt_instances[bi].gt_boxes
            if boxes.box_mode != BoxMode.XYXY_ABS:
                boxes = boxes.convert_mode(BoxMode.XYXY_ABS)
            boxes = boxes.tensor.cpu().numpy()
            ids = gt_instances[bi].gt_pids.cpu().tolist()
            for i in range(boxes.shape[0]):
                visualize_gt.draw_box(boxes[i])
                id_pos = boxes[i, :2]
                visualize_gt.draw_text(
                    str(ids[i]), id_pos, horizontal_alignment="left", color="w"
                )

            rgb_gt = visualize_gt.get_output().get_image()
            t_gt = tvF.to_tensor(rgb_gt)
            storage.put_image("img_{}/gt".format(bi), t_gt)

            visualize_preds = [
                Visualizer(img_rgb.copy()) for i in range(len(threds) - 1)
            ]
            boxes = pred_instances[bi].pred_boxes
            if boxes.box_mode != BoxMode.XYXY_ABS:
                boxes = boxes.convert_mode(BoxMode.XYXY_ABS)
            scores = pred_instances[bi].pred_scores.cpu().tolist()
            assign_ids = pred_instances[bi].assign_ids.cpu().tolist()
            for i, score in enumerate(scores):
                split = score_split(score, threds)
                b_clr = COLORS[i % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
                visualize_preds[split].draw_box(boxes[i], edge_color=b_clr)
                visualize_preds[split].draw_text(
                    "%.2f" % score,
                    boxes[i][2:],
                    horizontal_alignment="right",
                    color=t_clr,
                    bg_color=b_clr,
                )
                visualize_preds[split].draw_text(
                    str(assign_ids[i]),
                    boxes[i][:2],
                    horizontal_alignment="left",
                    color=t_clr,
                    bg_color=b_clr,
                )
            rgb_preds = [vis.get_output().get_image() for vis in visualize_preds]
            rgb_pred = np.concatenate(rgb_preds, axis=1)
            t_pred = tvF.to_tensor(rgb_pred)
            storage.put_image("img_{}/det".format(bi), t_pred)

    def vis_forward(self, image_list, gt_instances, pred_instances, feat_maps=None):
        raise NotImplementedError
