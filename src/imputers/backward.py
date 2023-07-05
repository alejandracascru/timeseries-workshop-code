from src.data.features import TIMESERIES_FEATURES
from .creation import imputers
from .imputer import Imputer
from .hardcoded import Hardcoded


class Backward(Imputer):
    def impute(self, data):
        common_columns = data.columns.intersection(TIMESERIES_FEATURES)
        data[common_columns] = data.groupby("patientunitstayid").apply(
            lambda group: group[common_columns].bfill()
        )

        data = Hardcoded().impute(data)

        return data


imputers.register_builder("backward", Backward)
