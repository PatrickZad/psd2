import sys

sys.path.append("./")
import fire
import torch
from psd2.checkpoint import DetectionCheckpointer as Checkpointer
from psd2.config import get_cfg
from psd2.modeling import build_model
from psd2.utils.events import EventStorage, TensorboardXWriter, get_event_storage
from psd2.utils.logger import setup_logger
import os


def build_train_loader(cfg):
    from psd2.data.catalog import MapperCatalog
    from psd2.data.build import build_detection_train_loader

    mapper = MapperCatalog.get(cfg.DATASETS.TRAIN[0])(cfg, is_train=False)
    return build_detection_train_loader(cfg, mapper=mapper)

def setup(config_file):
    setup_logger(name="psd2")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.VIS_PERIOD=1
    cfg.SOLVER.IMS_PER_BATCH=1


    device = torch.device("cuda")
    model = build_model(cfg)
    model.to(device)
    data_loader = build_train_loader(cfg=cfg)

    return cfg, model, data_loader,cfg.DATASETS.TEST[0]

def train_run(cfg, model, data_loader, dataset_name,writer):
    storage = get_event_storage()
    for batch_inputs in data_loader:
        model(batch_inputs)
        torch.cuda.empty_cache()
        writer.write()
        storage.step()


def main(cfg_file, resume=True):
    cfg, model, data_loader,dataset_name = setup(config_file=cfg_file)

    Checkpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume
    )
    writer = TensorboardXWriter(log_dir=cfg.OUTPUT_DIR)

    with EventStorage(start_iter=0):
        train_run(cfg, model, data_loader, dataset_name,writer)


if __name__ == "__main__":
    fire.Fire(main)
