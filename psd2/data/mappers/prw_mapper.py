from .mapper import SearchMapper, SearchMapperInfQuery, SearchMapperRE


class PrwMapper(SearchMapper):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)


class PrwMapperRE(SearchMapperRE):
    pass


class PrwSearchMapperInfQuery(SearchMapperInfQuery):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
