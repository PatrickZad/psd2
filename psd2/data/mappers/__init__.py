from .cuhk_sysu_mapper import CuhksysuMapper
from .prw_mapper import PrwMapper
from .mapper import SearchMapper
from .coco_ch_mapper import COCOCHDINOMapper, COCOCHDINOPreDetMapper, COCOCHMapper
from .movie_net_mapper import MovieNetMapper
from .cdps_mapper import CdpsMapper

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
