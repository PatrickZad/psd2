import sys

import numpy as np

sys.path.append("./")
import fire
from psd2.config import get_cfg
from psd2.data.build import get_detection_dataset_dicts
from psd2.data.catalog import MapperCatalog

from psd2.utils.visualizer import Visualizer
from psd2.utils.logger import setup_logger
from PIL import Image
import os
import torch
import torchvision.transforms.functional as tvF
from psd2.structures.boxes import BoxMode, Boxes


def setup_cfg(config_file):
    setup_logger(name="benchmark")
    setup_logger(name="data_test")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.SOLVER.IMS_PER_BATCH = 2

    return cfg


def test_dataset(cfg_file):
    cfg = setup_cfg(config_file=cfg_file)
    base_out = os.path.join(cfg.OUTPUT_DIR, "test", "data")
    dataset_name = cfg.DATASETS.TRAIN[0]
    cfg.OUTPUT_DIR = os.path.join(base_out, dataset_name)
    train_dicts = get_detection_dataset_dicts([dataset_name])
    vis_data_dict(train_dicts[:16], cfg)
    for dataset_name in cfg.DATASETS.TEST:
        cfg.OUTPUT_DIR = os.path.join(base_out, dataset_name)
        test_dicts = get_detection_dataset_dicts(dataset_name)
        vis_data_dict(test_dicts[:16], cfg)


def test_mapper(cfg_file):
    cfg = setup_cfg(config_file=cfg_file)
    base_out = os.path.join(cfg.OUTPUT_DIR, "test", "data_mapping")
    tr_dataset = cfg.DATASETS.TRAIN[0]
    train_dicts = get_detection_dataset_dicts(tr_dataset)
    train_mapper = MapperCatalog.get(tr_dataset)(cfg, is_train=True)
    cfg.OUTPUT_DIR = os.path.join(base_out, cfg.DATASETS.TRAIN[0])
    for ddict in train_dicts[:32]:
        mddict = train_mapper(ddict)
        vis_map_data([mddict], cfg)
    for test_set in cfg.DATASETS.TEST:
        mapper = MapperCatalog.get(test_set)(cfg, is_train=False)
        dataset_dict = get_detection_dataset_dicts(test_set)
        cfg.OUTPUT_DIR = os.path.join(base_out, test_set)
        for ddict in dataset_dict[:16]:
            mddict = mapper(ddict)
            vis_map_data([mddict], cfg)


def test_mapper_cococh(cfg_file):
    cfg = setup_cfg(config_file=cfg_file)
    base_out = os.path.join(cfg.OUTPUT_DIR, "test", "data_mapping")
    tr_dataset = cfg.DATASETS.TRAIN[0]
    train_dicts = get_detection_dataset_dicts(tr_dataset)
    train_mapper = MapperCatalog.get(tr_dataset)(cfg, is_train=True)
    cfg.OUTPUT_DIR = os.path.join(base_out, cfg.DATASETS.TRAIN[0])
    for ddict in train_dicts[:32]:
        mddict = train_mapper(ddict)
        mddict = mddict["global"] + mddict["local"]
        vis_map_data(mddict, cfg)


def vis_data_dict(data_dicts, cfg):
    save_dir = cfg.OUTPUT_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if "query" in data_dicts[0].keys():
        data_dicts = [ddict["query"] for ddict in data_dicts]
    for ddict in data_dicts:
        img = Image.open(ddict["file_name"])
        img_name = os.path.split(ddict["file_name"])[-1]
        img_vis = Visualizer(img)
        boxes = [ann["bbox"] for ann in ddict["annotations"]]
        box_modes = [ann["bbox_mode"] for ann in ddict["annotations"]]
        ids = [ann["person_id"] for ann in ddict["annotations"]]
        for box, box_mode, pid in zip(boxes, box_modes, ids):
            imgw, imgh = img.size
            box = (
                Boxes(box, box_mode)
                .convert_mode(BoxMode.XYXY_ABS, [imgh, imgw])
                .tensor[0]
                .numpy()
            )
            img_vis.draw_box(box)
            id_pos = box[:2]
            img_vis.draw_text(str(pid), id_pos, horizontal_alignment="left", color="w")
        img_vis.get_output().save(os.path.join(save_dir, img_name))


def vis_map_data(map_dicts, cfg):
    save_dir = cfg.OUTPUT_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if "query" in map_dicts[0].keys():
        data_dicts = []
        for ddict in map_dicts:
            data_dicts.append(ddict["query"])
    else:
        data_dicts = map_dicts
    for i, ddict in enumerate(data_dicts):
        img = ddict["image"].numpy().transpose(1, 2, 0) * 255
        gt_instances = ddict["instances"]
        img_name = os.path.split(gt_instances.file_name)[-1]
        img_vis = Visualizer(img)
        for box, pid in zip(
            gt_instances.gt_boxes.tensor.tolist(), gt_instances.gt_pids.tolist()
        ):
            img_vis.draw_box(box)
            id_pos = box[:2]
            img_vis.draw_text(str(pid), id_pos, horizontal_alignment="left", color="w")
        img_vis.get_output().save(os.path.join(save_dir, str(i) + "_" + img_name))


if __name__ == "__main__":
    fire.Fire()
