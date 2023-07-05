import os

import pandas as pd

from settings import ROOT_DIR
from .creation import datasets
from .eicu import EICU


class Mortality(EICU):
    def __init__(self, path, use_cat, use_num, encode_categorical):
        super().__init__(path, use_cat, use_num)
        self.encode_categorical = encode_categorical

        self.feature_columns = ["apacheadmissiondx", "ethnicity", "gender", "GCS Total", "Eyes", "Motor", "Verbal",
                                "admissionheight", "admissionweight", "age", "Heart Rate", "MAP (mmHg)",
                                "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation",
                                "Respiratory Rate", "Temperature (C)", "glucose", "FiO2", "pH"]

        self.cat_columns = ["apacheadmissiondx", "GCS Total"]
        self.num_columns = ["admissionheight", "admissionweight", "age", "Heart Rate", "MAP (mmHg)",
                            "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation",
                            "Respiratory Rate", "Temperature (C)", "glucose", "FiO2", "pH"]
        self.label_columns = ["hospitaldischargestatus"]

        self.raw_data = None
        self.load_data()

    @property
    def task(self):
        return "mortality"

    @property
    def num_outputs(self):
        return 2

    def load_data(self):
        data = pd.read_csv(os.path.join(ROOT_DIR, self.path))
        data = self._prepare_categorical_columns(data)
        data = self._filter_mortality_data(data)
        data[self.num_columns] = data[self.num_columns].astype("float32")

        self.raw_data = data

        labels = data[self.label_columns]
        data = data[self._get_used_columns()]

        data = self._encode_categorical_columns(data)
        data = pd.concat([data, labels], axis=1)

        self.data = data

    def _filter_mortality_data(self, data):
        data = data[data.gender != 0]  # actually okay since 0 means NaN in this case, not female
        data = data[data.hospitaldischargestatus != 2]
        data["unitdischargeoffset"] = data["unitdischargeoffset"] / (1440)
        data["itemoffsetday"] = (data["itemoffset"] / 24)
        data.drop(columns="itemoffsetday", inplace=True)
        mort_cols = ["patientunitstayid", "itemoffset", "apacheadmissiondx", "ethnicity", "gender",
                     "GCS Total", "Eyes", "Motor", "Verbal",
                     "admissionheight", "admissionweight", "age", "Heart Rate", "MAP (mmHg)",
                     "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation",
                     "Respiratory Rate", "Temperature (C)", "glucose", "FiO2", "pH",
                     "unitdischargeoffset", "hospitaldischargestatus"]

        all_mort = data[mort_cols]
        all_mort = all_mort[all_mort["unitdischargeoffset"] >= 2]
        all_mort = all_mort[all_mort["itemoffset"] > 0]

        return all_mort

    def _encode_categorical_columns(self, data):
        if self.use_cat and self.encode_categorical:
            data = pd.get_dummies(data, columns=self.cat_columns)

        return data

    def _get_used_columns(self):
        cols_used = ["patientunitstayid"]

        if self.use_num and self.use_cat:
            cols_used += self.cat_columns
            cols_used += self.num_columns
        elif self.use_num:
            cols_used += self.num_columns

        return cols_used


datasets.register_builder("mortality", Mortality)
