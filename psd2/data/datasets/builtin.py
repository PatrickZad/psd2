from psd2.data.datasets.cdps import load_cdps
from psd2.evaluation import evaluator
import os

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog, MetadataCatalog

from .cuhk_sysu import load_cuhk_sysu
from .prw import load_prw
from .movie_net import load_movie_net
from .ptk21 import load_ptk21
from .coco_ch import load_coco_ch, load_coco_p, load_cocofkp_ch,load_coco_ch_50,load_coco_ch_75,load_coco_ch_25
from .cpm import load_cpm
from .g2aps import load_g2a


# TODO change evaluator type to "query"
def register_cuhk_sysu_all(datadir):
    name = "CUHK-SYSU_" + "Train"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    name = "CUHK-SYSU_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_" + "GalleryGT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "CUHK-SYSU_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG50GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG100"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG100GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG100"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "CUHK-SYSU_" + "TestG4000GT"
    DatasetCatalog.register(name, lambda: load_cuhk_sysu(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")

def register_g2a_all(datadir):
    name = "G2APS_" + "Train"
    DatasetCatalog.register(name, lambda: load_g2a(datadir, "Train"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    name = "G2APS_" + "Gallery"
    DatasetCatalog.register(name, lambda: load_g2a(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "G2APS_" + "GalleryGT"
    DatasetCatalog.register(name, lambda: load_g2a(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "G2APS_" + "TestG50"
    DatasetCatalog.register(name, lambda: load_g2a(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "G2APS_" + "TestG50GT"
    DatasetCatalog.register(name, lambda: load_g2a(datadir, "TestG50"))
    MapperCatalog.register(name, mappers.CuhksysuMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")

def register_movie_net_all(datadir):
    name = "MovieNet_" + "Train_app10"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app10"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    name = "MovieNet_" + "Train_app30"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app30"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    name = "MovieNet_" + "Train_app50"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app50"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    name = "MovieNet_" + "Train_app70"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app70"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    name = "MovieNet_" + "Train_app100"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "Train_app100"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    name = "MovieNet_" + "GalleryTestG2000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG2000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "MovieNet_" + "GalleryTestG2000GT"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG2000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "MovieNet_" + "GalleryTestG4000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG4000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "MovieNet_" + "GalleryTestG4000GT"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG4000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "MovieNet_" + "GalleryTestG10000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG10000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "MovieNet_" + "GalleryTestG10000GT"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "GalleryTestG10000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "MovieNet_" + "TestG2000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG2000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "MovieNet_" + "TestG2000GT"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG2000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "MovieNet_" + "TestG4000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "MovieNet_" + "TestG4000GT"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG4000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "MovieNet_" + "TestG10000"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG10000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "MovieNet_" + "TestG10000GT"
    DatasetCatalog.register(name, lambda: load_movie_net(datadir, "TestG10000"))
    MapperCatalog.register(name, mappers.MovieNetMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")


def register_prw_all(datadir):
    name = "PRW_Train"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Train"))
    MapperCatalog.register(name, mappers.PrwMapper)
    name = "PRW_Query"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_QueryGT"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Query"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="query")
    name = "PRW_Gallery"
    DatasetCatalog.register(name, lambda: load_prw(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.PrwMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")
    name = "PRW_GalleryGT"
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

    name = "CDPS_Gallery"
    DatasetCatalog.register(name, lambda: load_cdps(datadir, "Gallery"))
    MapperCatalog.register(name, mappers.CdpsMapper)
    MetadataCatalog.get(name).set(evaluator_type="det")


def register_cococh_all(datadir):
    name = "COCO-CH"
    DatasetCatalog.register(
        name, lambda: load_coco_ch(datadir, "train", allow_crowd=False)
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-CH-50"
    DatasetCatalog.register(
        name, lambda: load_coco_ch_50(datadir, "train", allow_crowd=False)
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-CH-75"
    DatasetCatalog.register(
        name, lambda: load_coco_ch_75(datadir, "train", allow_crowd=False)
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-CH-25"
    DatasetCatalog.register(
        name, lambda: load_coco_ch_25(datadir, "train", allow_crowd=False)
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCOFKP-CH"
    DatasetCatalog.register(
        name, lambda: load_cocofkp_ch(datadir, "train", allow_crowd=False)
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)

    name = "COCO-CH-ML"
    DatasetCatalog.register(
        name,
        lambda: load_coco_ch(datadir, "train", allow_crowd=False, filter_esmall=True),
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-P"
    DatasetCatalog.register(
        name, lambda: load_coco_p(datadir, "train", allow_crowd=False)
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-P-ML"
    DatasetCatalog.register(
        name,
        lambda: load_coco_p(datadir, "train", allow_crowd=False, filter_esmall=True),
    )
    MapperCatalog.register(name, mappers.COCOCHMapper)
    name = "COCO-CH_DINO"
    DatasetCatalog.register(name, lambda: load_coco_ch(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCHDINOMapper)
    name = "COCO-P_DINO"
    DatasetCatalog.register(name, lambda: load_coco_p(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCHDINOMapper)
    name = "COCO-P_DINO_Stochastic"
    DatasetCatalog.register(name, lambda: load_coco_p(datadir, "train"))
    MapperCatalog.register(name, mappers.COCOCHDINOPreDetMapper)


def register_cpm(datadirs):
    name = "CPM_" + "Train"
    DatasetCatalog.register(name, lambda: load_cpm(datadirs, "Train"))
    MapperCatalog.register(name, mappers.SearchMapper)


_root = os.getenv("PS_DATASETS", "Data/ReID")
register_movie_net_all(os.path.join(_root, "movienet"))
register_cuhk_sysu_all(os.path.join(_root, "cuhk_sysu"))
register_g2a_all(os.path.join(_root, "G2APS/G2APS"))
register_prw_all(os.path.join(_root, "PRW"))
register_cdps_all(os.path.join(_root, "CDPS"))
register_cpm(
    [
        os.path.join(_root, "cuhk_sysu"),
        os.path.join(_root, "PRW"),
        os.path.join(_root, "movienet"),
    ]
)

_root_det = "Data/DetData"
register_cococh_all(_root_det)
