import os

import numpy as np
import pandas as pd

from settings import ROOT_DIR
from .creation import datasets
from .eicu import EICU


class Decompensation(EICU):
    def __init__(self, path, use_cat, use_num):
        super().__init__(path, use_cat, use_num)
        self.feature_columns = ["apacheadmissiondx", "ethnicity", "gender", "GCS Total", "Eyes", "Motor", "Verbal",
                                "admissionheight", "admissionweight", "age", "Heart Rate", "MAP (mmHg)",
                                "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation",
                                "Respiratory Rate", "Temperature (C)", "glucose", "FiO2", "pH"]
        self.cat_columns = ["apacheadmissiondx", "ethnicity", "gender", "GCS Total", "Eyes", "Motor", "Verbal"]
        self.num_columns = ["admissionheight", "admissionweight", "age", "Heart Rate", "MAP (mmHg)",
                            "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation",
                            "Respiratory Rate", "Temperature (C)", "glucose", "FiO2", "pH"]
        self.label_columns = ["unitdischargestatus"]
        self.load_data()

    @property
    def task(self):
        return "decompensation"

    @property
    def num_outputs(self):
        return 2

    def load_data(self):
        data = pd.read_csv(os.path.join(ROOT_DIR, self.path))
        data = self._prepare_categorical_columns(data)
        data = self._filter_decompensation_data(data)
        data = self._label_decompensation(data)
        data[self.num_columns] = data[self.num_columns].astype("float32")

        used_columns = self._get_used_columns()

        self.data = data[used_columns]

    def _filter_decompensation_data(self, data):
        dec_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'ethnicity', 'gender',
                    'GCS Total', 'Eyes', 'Motor', 'Verbal',
                    'admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
                    'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
                    'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH',
                    'unitdischargestatus']
        # all_df = all_df[all_df.gender != 0]
        # all_df = all_df[all_df.hospitaldischargestatus!=2]
        data['RLOS'] = np.nan
        data['unitdischargeoffset'] = data['unitdischargeoffset'] / (1440)
        data['itemoffsetday'] = (data['itemoffset'] / 24)
        data['RLOS'] = (data['unitdischargeoffset'] - data['itemoffsetday'])
        data.drop(columns='itemoffsetday', inplace=True)
        all_dec = data[data["unitdischargestatus"] != 2]
        all_dec = all_dec[all_dec['itemoffset'] > 0]
        all_dec = all_dec[(all_dec['unitdischargeoffset'] > 1) & (all_dec['RLOS'] > 0)]
        all_dec = all_dec[dec_cols]

        return all_dec

    def _label_decompensation(self, data):
        data["temp_y"] = np.nan
        data["temp_y"] = data["itemoffset"] - 48
        data['count_max'] = data.groupby(['patientunitstayid'])['temp_y'].transform(max)
        data["label_24"] = np.nan
        data.loc[data['itemoffset'] < data['count_max'], "label_24"] = 0
        data.loc[data['itemoffset'] >= data['count_max'], "label_24"] = data['unitdischargestatus']
        data["unitdischargestatus"] = data["label_24"]
        data.drop(columns=['temp_y', 'count_max', 'label_24'], inplace=True)
        data.unitdischargestatus = data.unitdischargestatus.astype(int)

        return data

    def _get_used_columns(self):
        cols_used = ["patientunitstayid"]

        if self.use_num and self.use_cat:
            cols_used += self.cat_columns
            cols_used += self.num_columns
            cols_used += ["unitdischargestatus"]

        elif self.use_num:
            cols_used += self.num_columns
            cols_used += ["unitdischargestatus"]

        return cols_used


datasets.register_builder("decompensation", Decompensation)
