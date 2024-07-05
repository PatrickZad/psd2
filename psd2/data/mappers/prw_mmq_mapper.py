from .mapper import SearchMapperMmq

class PrwMmqMapper(SearchMapperMmq):
    def __init__(self, cfg, is_train) -> None:
        super().__init__(cfg, is_train)
