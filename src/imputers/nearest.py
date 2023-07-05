from .creation import imputers
from .imputer import Imputer


class Nearest(Imputer):
    def impute(self, data):
        data = data.groupby(['patientunitstayid']).apply(
            lambda group: group.interpolate(method='nearest')
        )
        return data


imputers.register_builder("nearest", Nearest)
