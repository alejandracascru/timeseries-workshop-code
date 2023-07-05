import copy

import numpy as np
from pyod.models.knn import KNN as PyodKNN

from src.data.features import STATIC_FEATURES, TIMESERIES_FEATURES
from .creation import outlier_removers
from .outlier_remover import OutlierRemover


class KNN(OutlierRemover):
    def remove(self, data):
        data = copy.deepcopy(data)
        od_model = PyodKNN()

        # Remove subjects with outlier static features
        static_columns = data.columns.intersection(STATIC_FEATURES)

        if len(static_columns) > 0:
            data = data.drop(index=data.index[np.where(data[static_columns].isna().any(axis=1))[0]])
            od_model.fit(np.nan_to_num(data[static_columns].to_numpy()))
            data = data.drop(index=data.index[np.where(od_model.labels_ == 1)[0]])

        # Replace time series features with Nan for outlier records
        timeseries_columns = data.columns.intersection(TIMESERIES_FEATURES)
        od_model.fit(np.nan_to_num(data[timeseries_columns].to_numpy()))
        data.loc[od_model.labels_ == 1, timeseries_columns] = np.nan

        return data


outlier_removers.register_builder("knn", KNN)
