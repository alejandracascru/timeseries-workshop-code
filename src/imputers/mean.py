from src.data.features import TIMESERIES_FEATURES
from .creation import imputers
from .hardcoded import Hardcoded
from .imputer import Imputer


class Mean(Imputer):
    def impute(self, data):
        common_columns = data.columns.intersection(TIMESERIES_FEATURES)
        data[common_columns] = data.groupby(['patientunitstayid'], group_keys=False).apply(
            lambda group: group[common_columns].fillna(group[common_columns].mean())
        )

        data = Hardcoded().impute(data)

        return data


imputers.register_builder("mean", Mean)
