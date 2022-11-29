from .mapper import SearchMapper, SearchMapperInfQuery,SearchMapperRE


class CuhksysuMapper(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)

class CuhksysuMapperRE(SearchMapperRE):
    pass

class CuhkSearchMapperInfQuery(SearchMapperInfQuery):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
