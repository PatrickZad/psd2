import sys
import numpy as np
sys.path.append("./")
import fire
from psd2.config import get_cfg
from psd2.data.build import get_detection_dataset_dicts
from psd2.utils.logger import setup_logger
from PIL import Image
import os
from psd2.structures.boxes import BoxMode, Boxes
import matplotlib.pyplot as plt
import seaborn as sns
import io
import cv2
import pandas as pd
sns.set(font_scale=1.0,style="white")
def setup_cfg(config_file):
    setup_logger(name="benchmark")
    setup_logger(name="data_test")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.SOLVER.IMS_PER_BATCH = 2

    return cfg
def scale_hist(cfg_file):
    cfg = setup_cfg(config_file=cfg_file)
    base_out = os.path.join(cfg.OUTPUT_DIR, "stata", "scale")
    dataset_name = cfg.DATASETS.TRAIN[0]
    cfg.OUTPUT_DIR = os.path.join(base_out, dataset_name)
    train_dicts = get_detection_dataset_dicts([dataset_name])
    id_area_dict={}
    for ddict in train_dicts:
        img = Image.open(ddict["file_name"])
        boxes = [ann["bbox"] for ann in ddict["annotations"]]
        box_modes = [ann["bbox_mode"] for ann in ddict["annotations"]]
        ids = [ann["person_id"] for ann in ddict["annotations"]]
        for box, box_mode, pid in zip(boxes, box_modes, ids):
            imgw, imgh = img.size
            area = (
                Boxes(box, box_mode)
                .convert_mode(BoxMode.XYXY_ABS, [imgh, imgw])
                .area()[0]
                .numpy()
            )
            if pid >-1:
                if pid in id_area_dict:
                    id_area_dict[pid].append(area)
                else:
                    id_area_dict[pid]=[area]
    test_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST[0])
    for ddict in test_dicts:
        img = Image.open(ddict["file_name"])
        boxes = [ann["bbox"] for ann in ddict["annotations"]]
        box_modes = [ann["bbox_mode"] for ann in ddict["annotations"]]
        ids = [ann["person_id"] for ann in ddict["annotations"]]
        for box, box_mode, pid in zip(boxes, box_modes, ids):
            imgw, imgh = img.size
            area = (
                Boxes(box, box_mode)
                .convert_mode(BoxMode.XYXY_ABS, [imgh, imgw])
                .area()[0]
                .numpy()
            )
            if pid >-1:
                if pid in id_area_dict:
                    id_area_dict[pid].append(area)
                else:
                    id_area_dict[pid]=[area]
    ratios=[]
    for pid,areas in id_area_dict.items():
        a_arr=np.array(areas)
        ratio=a_arr.max()/a_arr.min()
        if ratio <=10.0:
            ratios.append("1.0-10.0")
        elif ratio<=20.0:
            ratios.append("10.0-20.0")
        elif ratio<=30.0:
            ratios.append("20.0-30.0")
        elif ratio<=40.0:
            ratios.append("30.0-40.0")
        elif ratio<=50.0:
            ratios.append("40.0-50.0")
        else:
            ratios.append(">50.0")
    pal=sns.color_palette("hls",6)
    hisimg=_vis_hist(ratios,"Area ratio",["1.0-10.0","10.0-20.0","20.0-30.0","30.0-40.0","40.0-50.0",">50.0"],pal)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"scale_ratios.png"),hisimg)
def scale_hist_cuhk(cfg_file):
    cfg = setup_cfg(config_file=cfg_file)
    base_out = os.path.join(cfg.OUTPUT_DIR, "stata", "scale")
    dataset_name = cfg.DATASETS.TRAIN[0]
    cfg.OUTPUT_DIR = os.path.join(base_out, dataset_name)
    train_dicts = get_detection_dataset_dicts([dataset_name])
    id_area_dict={}
    for ddict in train_dicts:
        img = Image.open(ddict["file_name"])
        boxes = [ann["bbox"] for ann in ddict["annotations"]]
        box_modes = [ann["bbox_mode"] for ann in ddict["annotations"]]
        ids = [ann["person_id"] for ann in ddict["annotations"]]
        for box, box_mode, pid in zip(boxes, box_modes, ids):
            imgw, imgh = img.size
            area = (
                Boxes(box, box_mode)
                .convert_mode(BoxMode.XYXY_ABS, [imgh, imgw])
                .area()[0]
                .numpy()
            )
            if pid >-1:
                if pid in id_area_dict:
                    id_area_dict[pid].append(area)
                else:
                    id_area_dict[pid]=[area]
    test_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST[0])
    for ddict in test_dicts:
        img = Image.open(ddict["file_name"])
        boxes = [ann["bbox"] for ann in ddict["annotations"]]
        box_modes = [ann["bbox_mode"] for ann in ddict["annotations"]]
        ids = [ann["person_id"] for ann in ddict["annotations"]]
        for box, box_mode, pid in zip(boxes, box_modes, ids):
            imgw, imgh = img.size
            area = (
                Boxes(box, box_mode)
                .convert_mode(BoxMode.XYXY_ABS, [imgh, imgw])
                .area()[0]
                .numpy()
            )
            if pid >-1:
                if pid in id_area_dict:
                    id_area_dict[pid].append(area)
                else:
                    id_area_dict[pid]=[area]
    ratios=[]
    for pid,areas in id_area_dict.items():
        a_arr=np.array(areas)
        ratio=a_arr.max()/a_arr.min()
        if ratio <=2.0:
            ratios.append("1.0-2.0")
        elif ratio<=3.0:
            ratios.append("2.0-3.0")
        elif ratio<=4.0:
            ratios.append("3.0-4.0")
        elif ratio<=5.0:
            ratios.append("4.0-5.0")
        else:
            ratios.append(">5.0")
    pal=sns.color_palette("hls",5)
    hisimg=_vis_hist(ratios,"Area ratio",["1.0-2.0","2.0-3.0","3.0-4.0","4.0-5.0",">5.0"],pal)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"scale_ratios.png"),hisimg)
def samples_hist(cfg_file):
    cfg = setup_cfg(config_file=cfg_file)
    base_out = os.path.join(cfg.OUTPUT_DIR, "stata", "scale")
    dataset_name = cfg.DATASETS.TRAIN[0]
    cfg.OUTPUT_DIR = os.path.join(base_out, dataset_name)
    train_dicts = get_detection_dataset_dicts([dataset_name])
    id_nsamp_dict={}
    n_ulb=0
    for ddict in train_dicts:
        ids = [ann["person_id"] for ann in ddict["annotations"]]
        for pid in ids:
            if pid >-1:
                if pid in id_nsamp_dict:
                    id_nsamp_dict[pid]+=1
                else:
                    id_nsamp_dict[pid]=1
            else:
                n_ulb+=1
    n_samples=list(id_nsamp_dict.values())
    n_samples=sorted(n_samples,reverse=True)
    vis_data = pd.DataFrame(
            {
                "Sorted person identity": list(range(len(n_samples))),
                "Number of training samples": n_samples,
                "type": ["n"] * len(n_samples)
            },
    )
    fig = plt.figure()
    sns.lineplot(data=vis_data,x="Sorted person identity",y="Number of training samples",hue="type",legend=False,palette=['r',])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"sample_numbers.png"),img)
    print("number ulb: {}".format(n_ulb))
def _vis_hist(data,name,order,colors):
    vis_data = pd.DataFrame(
            {
                name: data,
                "type": data,
            },
    )
    vis_data[name]=pd.Categorical(vis_data[name],order)
    fig = plt.figure()
    nbins=10
    ax=sns.histplot(
            data=vis_data,
            x=name,
            hue="type",
            stat="percent",
            legend=False,
            palette=colors,
            discrete= True,
            # order = order,
    )
    for p in ax.patches:
        value=float("{:.2f}".format(p.get_height()))
        if value == 0.0:
            continue
        ax.text(p.get_x() + p.get_width() / 2., p.get_height()-1, value , 
                fontsize=12, ha='center', va='top')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if __name__ == "__main__":
    fire.Fire()