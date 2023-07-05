import os

import pandas as pd

from settings import ROOT_DIR
from .creation import datasets
from .eicu import EICU


class Forecasting(EICU):
    def __init__(self, path, feature, use_other_features, patient_id, cutoff):
        super().__init__(path, None, None)

        self.num_columns = ["admissionheight", "admissionweight", "age", "Heart Rate", "MAP (mmHg)",
                            "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation",
                            "Respiratory Rate", "Temperature (C)", "glucose", "FiO2", "pH"]
        self.feature = feature
        self.use_other_features = use_other_features
        self.patient_id = patient_id
        self.cutoff = cutoff
        self.load_data()

    @property
    def task(self):
        return "forecasting"

    @property
    def num_outputs(self):
        return 1

    @property
    def num_features(self):
        return self.data.shape[1] - 1

    def load_data(self):
        data = pd.read_csv(os.path.join(ROOT_DIR, self.path))
        data[self.feature] = data[self.feature].astype("float32")
        data = self._filter_forecasting_data(data)
        data = self._create_cutoff(data)

        data = data[self._get_used_columns()]

        target = data[self.feature]
        data = data[data.columns[~data.columns.isin([self.feature])]]

        data = pd.concat([data, target], axis=1)
        self.data = data

    def _get_used_columns(self):
        cols_used = ["patientunitstayid"]

        if self.use_other_features:
            cols_used += self.num_columns
        else:
            cols_used += [self.feature]

        return cols_used

    def _filter_forecasting_data(self, data):
        data = data.loc[data["itemoffset"] >= 0]
        data = data.loc[data["patientunitstayid"] == self.patient_id]

        return data

    def _create_cutoff(self, data):
        num_timesteps = len(data)
        cutoff = int(self.cutoff * num_timesteps)
        cutoff = data.iloc[cutoff]["itemoffset"]

        data.loc[data["itemoffset"] <= cutoff, "patientunitstayid"] = 1
        data.loc[data["itemoffset"] > cutoff, "patientunitstayid"] = 2

        return data


datasets.register_builder("forecasting", Forecasting)
