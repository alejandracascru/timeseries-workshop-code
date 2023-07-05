from src.data.features import TIMESERIES_FEATURES
from .creation import imputers
from .hardcoded import Hardcoded
from .imputer import Imputer


class Polynomial(Imputer):
    def impute(self, data):
        common_columns = data.columns.intersection(TIMESERIES_FEATURES)
        data[common_columns] = data.groupby(['patientunitstayid'], group_keys=False).apply(
            lambda group: group[common_columns] if (group[common_columns].count() < 4).any() else
            group[common_columns].interpolate(method="polynomial", limit_area="inside",
                                       limit_direction="both", order=3)
        )

        data = Hardcoded().impute(data)

        return data


imputers.register_builder("polynomial", Polynomial)
