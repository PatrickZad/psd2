import psd2.data.transforms as dT
import torchvision.transforms as tT

import numpy as np

from ..detection_utils import read_image
from copy import deepcopy

import numbers
import warnings

import math
from psd2.structures.instances import Instances
from psd2.structures.boxes import Boxes
import torch


class SearchMapper(object):
    def __init__(self, cfg, is_train) -> None:
        self.is_train = is_train
        self.in_fmt = cfg.INPUT.FORMAT
        img_mean = cfg.MODEL.PIXEL_MEAN
        img_std = cfg.MODEL.PIXEL_STD
        if max(img_mean) > 1:  # deal with mean and std, to_tensor makes div(255)
            img_mean = np.array(img_mean) / 255.0
            img_mean = img_mean.tolist()
            img_std = np.array(img_std) / 255.0
            img_std = img_std.tolist()
        if self.is_train:
            self.augs = dT.AugmentationList(
                [
                    dT.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                        sample_style="choice",
                    ),
                    dT.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                ]
            )
        else:
            self.augs = dT.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                sample_style="choice",
            )
        self.totensor_norm = tT.Compose(
            [
                tT.ToTensor(),
                tT.Normalize(mean=img_mean, std=img_std),
            ]
        )
        if cfg.INPUT.REA.ENABLED:
            self.rea = RandomInstanceErasingNormTensor(
                pix_mean=img_mean, pix_std=img_std
            )
        else:
            self.rea = trivial_rea

    def __call__(self, img_dict):
        """
        For query, img_dict is
        {
            "query":
                {
                    "file_name": image paths,
                    "image_id": image name,
                    "image": augmented image tensor,
                    "width","height",
                    "boxes": augmented boxes,
                    "ids","org_width","org_height","org_boxes",
                    "annotations":
                        [
                            {
                                "bbox": person xyxy_abs boxes,
                                "bbox_mode": format of bbox
                                "person_id":  person id
                            }
                        ],
                },
            "gallery": (Optional)
                [
                    {
                        "file_name": image paths,
                        "image_id": image name,
                        "annotations":
                            [
                                {
                                    "bbox": person xyxy_abs boxes,
                                    "bbox_mode": format of bbox,                                                "person_id":  person id
                                }
                            ],
                    },
                    ...
                ]
        }
        """
        if "query" in img_dict.keys():
            rst = {}
            rst["query"] = self._transform_annotations(img_dict["query"])
            if "gallery" in img_dict:
                rst["gallery"] = []
                for gann in img_dict["gallery"]:
                    g_instance = Instances(
                        image_size=None,
                        file_name=gann["file_name"],
                        image_id=gann["image_id"],
                        gt_boxes=Boxes(gann["bbox"]),
                    )
                    rst["gallery"].append(g_instance)
            return rst
        else:
            return self._transform_annotations(img_dict)

    def _transform_annotations(self, img_dict):
        img_path = img_dict["file_name"]
        img_arr = read_image(img_path, self.in_fmt)
        boxes = []
        ids = []
        for ann in img_dict["annotations"]:
            boxes.append(ann["bbox"])
            ids.append(ann["person_id"])
        org_boxes = np.array(boxes, dtype=np.float32)
        aug_input = dT.AugInput(image=img_arr.copy(), boxes=org_boxes.copy())
        transforms = self.augs(aug_input)
        aug_img = aug_input.image
        h, w = aug_img.shape[:2]
        aug_boxes = aug_input.boxes
        img_t = self.totensor_norm(aug_img.copy())
        self.rea(img_t)
        return {
            "image": img_t,
            "instances": Instances(
                (h, w),
                file_name=img_path,
                image_id=img_dict["image_id"],
                gt_boxes=Boxes(aug_boxes),
                gt_pids=torch.tensor(ids, dtype=torch.int),
                gt_classes=torch.zeros(len(ids), dtype=torch.int),
                org_img_size=(img_arr.shape[0], img_arr.shape[1]),
                org_gt_boxes=Boxes(org_boxes),
            ),
        }


class RandomInstanceErasingNormTensor:
    """Randomly selects a rectangle region in a person and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against bounding box.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.

    Returns:
        Erased Image.

    """

    def __init__(
        self,
        pix_mean,
        pix_std,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=[0.485, 0.456, 0.406],
    ):
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence"
            )
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = torch.as_tensor(scale)
        self.ratio = torch.as_tensor(ratio)
        self.value = (
            torch.as_tensor(value) - torch.as_tensor(pix_mean)
        ) / torch.as_tensor(pix_std)

    def get_params(self, box, channels, scale, ratio, value=None):
        """Get parameters for ``erase`` for a random erasing."""
        box = torch.round(box).astype(torch.int32)
        img_c, img_h, img_w = channels, box[3] - box[1], box[2] - box[0]
        area = img_h * img_w

        log_ratio = torch.log(ratio)
        for _ in range(10):

            erase_area = (
                area * (torch.rand(1) * (scale[1] - scale[0]) + scale[0]).item()
            )
            aspect_ratio = torch.exp(
                (torch.rand(1) * (log_ratio[1] - log_ratio[0]) + log_ratio[0])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.normal(0.0, 1.0, size=(img_c, h, w))
            else:
                v = value.view(img_c, 1, 1)

            i = torch.randint(0, img_w - w + 1, size=1).item()
            j = torch.randint(0, img_h - h + 1, size=1).item()
            return i + box[0].item(), j + box[1].item(), h, w, v

        # Return original image
        return box[0].item(), box[1].item(), 0, 0, 0

    def __call__(self, img, boxes):
        vals = torch.rand(boxes.shape[0])
        for i in range(boxes.shape[0]):
            if vals[i] < self.p:
                x, y, h, w, v = self.get_params(
                    boxes[i], 3, scale=self.scale, ratio=self.ratio, value=self.value
                )
                self.erase(img, x, y, h, w, v)

    def erase(self, img, i, j, h, w, v):

        img[j : j + h, i : i + w] = v


def trivial_rea(img, ins_boxes):
    pass
