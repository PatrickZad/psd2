import sys
sys.path.append("./")
import os
import torch
import fire
from psd2.config import get_cfg
from psd2.checkpoint import DetectionCheckpointer as Checkpointer
from psd2.modeling import build_model
from psd2.utils.events import EventStorage, TensorboardXWriter, get_event_storage
from psd2.utils.logger import setup_logger

def build_test_loader(cfg, dataset_name):
    from psd2.data.catalog import MapperCatalog
    from psd2.data.build import build_detection_test_loader

    mapper = MapperCatalog.get(dataset_name)(cfg, False)
    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup(config_file):
    setup_logger(name="psd2")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.VIS_PERIOD = 1

    device = torch.device("cuda")
    model = build_model(cfg)
    model.to(device)
    data_loader = build_test_loader(cfg=cfg, dataset_name=cfg.DATASETS.TEST[0])
    dataset_name=cfg.DATASETS.TEST[0]


    return cfg, model, data_loader,dataset_name



def test_vis(cfg, model, data_loader, writer):
    model.train()
    storage = get_event_storage()
    for batch_inputs in data_loader:
        out = model.forward(batch_inputs)
        torch.cuda.empty_cache()
        writer.write()
        storage.step()


def main(cfg_file, resume=False):
    cfg, model, data_loader,dataset_name = setup(config_file=cfg_file)

    Checkpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume
    )
    cfg.OUTPUT_DIR=cfg.OUTPUT_DIR+"/visualize/"+dataset_name

    writer = TensorboardXWriter(log_dir=cfg.OUTPUT_DIR)

    with EventStorage(start_iter=0):
        test_vis(cfg, model, data_loader, writer)


if __name__ == "__main__":
    fire.Fire(main)
