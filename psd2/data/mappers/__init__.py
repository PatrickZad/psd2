from .cuhk_sysu_mapper import CuhksysuMapper
from .prw_mapper import PrwMapper
from .mapper import SearchMapper

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
