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



def build_test_loader( cfg, dataset_name):
        from psd2.data.catalog import MapperCatalog
        from psd2.data.build import get_detection_dataset_dicts, trivial_batch_collator
        from psd2.data.common import DatasetFromList, MapDataset
        from psd2.data.samplers import InferenceSampler
        import torch.utils.data as torchdata

        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        mapper = MapperCatalog.get(dataset_name)(cfg, is_train=False)
        # batched test
        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False)
        dataset = MapDataset(dataset, mapper)
        sampler = InferenceSampler(len(dataset))

        batch_sampler = torchdata.sampler.BatchSampler(
            sampler, cfg.TEST.IMS_PER_PROC, drop_last=False
        )
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader

def setup(config_file):
    setup_logger(name="psd2")

    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    device = torch.device("cuda")
    model = build_model(cfg)
    model.to(device)
    data_loader = build_test_loader(cfg=cfg, dataset_name=cfg.DATASETS.TEST[0])

    return cfg, model, data_loader,cfg.DATASETS.TEST[0]

def save_inference(cfg, model, data_loader, dataset_name,writer):
    model.eval()
    storage = get_event_storage()
    rst=[]
    for batch_inputs in data_loader:
        out = model(batch_inputs)
        rst.append(out)
        torch.cuda.empty_cache()
        writer.write()
        storage.step()
    rst=torch.cat(rst,dim=0)
    torch.save(rst,"{}/{}_img_feats.pth".format(cfg.OUTPUT_DIR,dataset_name))


def main(cfg_file, resume=False):
    cfg, model, data_loader,dataset_name = setup(config_file=cfg_file)

    Checkpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume
    )
    writer = TensorboardXWriter(log_dir=cfg.OUTPUT_DIR)

    with EventStorage(start_iter=0):
        save_inference(cfg, model, data_loader, dataset_name,writer)


if __name__ == "__main__":
    fire.Fire(main)
