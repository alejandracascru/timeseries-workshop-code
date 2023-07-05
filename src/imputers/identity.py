from .creation import imputers
from .imputer import Imputer


class Identity(Imputer):
    def impute(self, data):
        return data


imputers.register_builder("identity", Identity)
