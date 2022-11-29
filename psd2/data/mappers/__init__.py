from .cuhk_sysu_mapper import CuhksysuMapper, CuhkSearchMapperInfQuery, CuhksysuMapperRE
from .prw_mapper import PrwMapper, PrwSearchMapperInfQuery, PrwMapperRE
from .mapper import SearchMapper, SearchMapperInfQuery
from .cdps_mapper import *
from .ptk21_mapper import *
from .coco_ch_mapper import COCOCHMapper, COCOCH2vMapper

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
