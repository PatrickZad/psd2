import psd2.data.transforms as dT
import torchvision.transforms as tT
import torchvision.transforms.functional as tvtF
from ..detection_utils import read_image
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from psd2.structures import Instances, Boxes, BoxMode
import torch
from .mapper import SearchMapper

class COCOCHMapper(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)


class COCOCHDINOMapper(object):
    # 2 for global, 1 for local, as DINO does
    # TODO consider instance-wise aug
    def __init__(self, cfg, is_train) -> None:
        assert is_train
        self.is_train = is_train
        img_mean = cfg.MODEL.PIXEL_MEAN
        img_std = cfg.MODEL.PIXEL_STD
        self.num_local_view = cfg.PERSON_SEARCH.DINO.NUM_LOCAL_TOKEN_GROUPS
        self.augs_box1 = dT.AugmentationList(
            [
                dT.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                    sample_style="choice",
                    interp=Image.BICUBIC,
                ),
                dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
        )
        self.augs_box2 = dT.AugmentationList(
            [
                dT.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                    sample_style="choice",
                    interp=Image.BICUBIC,
                ),
                dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
        )
        self.augs_box_local = dT.AugmentationList(
            [
                dT.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                    sample_style="choice",
                    interp=Image.BICUBIC,
                ),
                dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
        )
        color_jitter = tT.Compose(
            [
                tT.RandomApply(
                    [
                        tT.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                tT.RandomGrayscale(p=0.2),
            ]
        )
        self.augs_color1 = tT.transforms.Compose([color_jitter, GaussianBlur(1.0)])
        self.augs_color2 = tT.transforms.Compose(
            [color_jitter, GaussianBlur(0.1), Solarization(0.2)]
        )
        self.augs_color_local = tT.transforms.Compose([color_jitter, GaussianBlur(0.5)])
        self.norm = tT.Normalize(mean=img_mean, std=img_std)

        self.to_pil = tT.ToPILImage()
        self.img_fmt = cfg.INPUT.FORMAT

    def __call__(self, img_dict):
        return self._common_map(img_dict)

    def _common_map(self, img_dict):
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.img_fmt)
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            box_mode = ann["bbox_mode"]
            boxes.append(
                Boxes(ann["bbox"], box_mode)
                .convert_mode(BoxMode.XYXY_ABS, img_arr.shape[:2])
                .tensor[0]
                .tolist()
            )
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        # img 1
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs_box1(aug_input)
        aug_img = aug_input.image
        aug_boxes = aug_input.boxes
        aug_h, aug_w = aug_img.shape[0], aug_img.shape[1]
        aug_img = self.augs_color1(self.to_pil(aug_img))
        imgt_1 = tvtF.to_tensor(aug_img.copy())
        img_1_inst = {
            "image": imgt_1,
            "instances": Instances(
                (aug_h, aug_w),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes, BoxMode.XYXY_ABS),
                gt_pids=torch.tensor(ids, dtype=torch.int64),
                gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
            ),
        }
        # img 2
        aug_input_2 = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms_2 = self.augs_box2(aug_input_2)
        aug_img_2 = aug_input_2.image
        aug_h_2, aug_w_2 = aug_img_2.shape[0], aug_img_2.shape[1]
        aug_boxes_2 = aug_input_2.boxes
        aug_img_2 = self.augs_color2(self.to_pil(aug_img_2))
        imgt_2 = tvtF.to_tensor(aug_img_2.copy())
        img_2_inst = {
            "image": imgt_2,
            "instances": Instances(
                (aug_h_2, aug_w_2),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes_2, BoxMode.XYXY_ABS),
                gt_pids=torch.tensor(ids, dtype=torch.int64),
                gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
            ),
        }
        # local views
        img_insts = []
        for i in range(self.num_local_view):
            aug_input_local = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
            transforms = self.augs_box_local(aug_input_local)
            aug_img_local = aug_input_local.image
            aug_h_local, aug_w_local = aug_img_local.shape[0], aug_img_local.shape[1]
            aug_boxes_local = aug_input_local.boxes
            aug_img_local = self.augs_color_local(self.to_pil(aug_img_local))
            imgt_local = tvtF.to_tensor(aug_img_local.copy())
            img_inst_local = {
                "image": imgt_local,
                "instances": Instances(
                    (aug_h_local, aug_w_local),
                    file_name=img_path,
                    image_id=img_dict["image_id"],
                    gt_boxes=Boxes(aug_boxes_local, BoxMode.XYXY_ABS),
                    gt_pids=torch.tensor(ids, dtype=torch.int64),
                    gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                    org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                    org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
                ),
            }
            img_insts.append(img_inst_local)

        return {"global": [img_1_inst, img_2_inst], "local": img_insts}


class COCOCHDINOPreDetMapper(object):
    # 1 for global, 1 for local, as DINO does
    # TODO consider instance-wise aug
    def __init__(self, cfg, is_train) -> None:
        assert is_train
        self.is_train = is_train
        img_mean = cfg.MODEL.PIXEL_MEAN
        img_std = cfg.MODEL.PIXEL_STD
        self.augs_box = dT.AugmentationList(
            [
                dT.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                    sample_style="choice",
                    interp=Image.BICUBIC,
                ),
                dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
        )
        self.norm = tT.Normalize(mean=img_mean, std=img_std)

        self.to_pil = tT.ToPILImage()
        self.img_fmt = cfg.INPUT.FORMAT

    def __call__(self, img_dict):
        return self._common_map(img_dict)

    def _common_map(self, img_dict):
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.img_fmt)
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            box_mode = ann["bbox_mode"]
            boxes.append(
                Boxes(ann["bbox"], box_mode)
                .convert_mode(BoxMode.XYXY_ABS, img_arr.shape[:2])
                .tensor[0]
                .tolist()
            )
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs_box(aug_input)
        aug_img = aug_input.image
        aug_boxes = aug_input.boxes
        aug_h, aug_w = aug_img.shape[0], aug_img.shape[1]
        imgt = tvtF.to_tensor(aug_img.copy())
        return {
            "image": imgt,
            "instances": Instances(
                (aug_h, aug_w),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes, BoxMode.XYXY_ABS),
                gt_pids=torch.tensor(ids, dtype=torch.int64),
                gt_classes=torch.zeros(len(ids), dtype=torch.int64),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes, BoxMode.XYXY_ABS),
            ),
        }


import random


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
