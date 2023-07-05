import copy

from sklearn.preprocessing import StandardScaler

from .creation import normalizers
from .normalizer import Normalizer


class Standardization(Normalizer):
    def __init__(self):
        self.normalize_columns = ["admissionheight", "admissionweight", "age", "Heart Rate", "MAP (mmHg)",
                                  "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation",
                                  "Respiratory Rate", "Temperature (C)", "glucose", "FiO2", "pH"]
        self.scaler = None

    def fit(self, data):
        scaler = StandardScaler()
        common_columns = set(data.columns).intersection(set(self.normalize_columns))
        features = data[common_columns]
        scaler.fit(features.values)

        self.scaler = scaler

    def transform(self, data):
        data = copy.deepcopy(data)
        common_columns = set(data.columns).intersection(set(self.normalize_columns))
        features = data[common_columns]
        data[list(common_columns)] = self.scaler.transform(features.values)

        return data

normalizers.register_builder("standardization", Standardization)