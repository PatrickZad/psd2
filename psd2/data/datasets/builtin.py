from psd2.data.datasets.cdps import load_cdps
from psd2.evaluation import evaluator
import os

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog, MetadataCatalog

from .cuhk_sysu import load_cuhk_sysu
from .cuhk_sysu import subset_names as cuhk_subsets
from .prw import load_prw
from .prw import subsets as prw_subsets
from .ptk21 import load_ptk21
from .coco_ch import load_coco_ch, load_coco_p
import copy

# TODO change evaluator type to "query"
def register_cuhk_sysu_all(datadir):
    name = "CUHK-SYSU_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    name = "CUHK-SYSU_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")


def register_prw_all(datadir):

    name = "PRW_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwMapper)
    name = "PRW_Query"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_Gallery"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


def register_ptk21_all(datadir):
    name = "PoseTrack21_Train"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Train"))
    MapperCatalog.register(name, mappers.Ptk21Mapper)
    # MetadataCatalog.get(name).set(evaluator_type="det")
    name = "PoseTrack21_Query"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Query"))
    MapperCatalog.register(name, mappers.Ptk21Mapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PoseTrack21_InfQ"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Query"))
    MapperCatalog.register(name, mappers.Ptk21SearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PoseTrack21_Gallery"
    DatasetCatalog.register(name, lambda: load_ptk21(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.Ptk21Mapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


def register_cdps_all(datadir):
    name = "CDPS_Train"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Train"))
    MapperCatalog.register(name, mappers.CdpsMapper)
    # MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CDPS_Query"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Query"))
    MapperCatalog.register(name, mappers.CdpsMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CDPS_InfQ"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Query"))
    MapperCatalog.register(name, mappers.CdpsSearchMapperInfQuery)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CDPS_Gallery"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CdpsMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


def register_cococh_all(datadir):
    name = "COCO-CH_DINO"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCHDINOMapper)
    name = "COCO-P_DINO"
    DatasetCatalog.register(name, lambda: load_coco_p(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCHDINOMapper)
    name = "COCO-P_DINO_Stochastic"
    DatasetCatalog.register(name, lambda: load_coco_p(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCHDINOPreDetMapper)


_root = os.getenv("PS_DATASETS", "Data/ReID")
register_cuhk_sysu_all(os.path.join(_root, "cuhk_sysu"))
register_prw_all(os.path.join(_root, "PRW"))
_root_det = "Data/DetData"
register_cococh_all(_root_det)
