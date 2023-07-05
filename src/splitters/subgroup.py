import numpy as np

from .creation import splitters
from .splitter import Splitter


feature_mapping = {"ethnicity": {'Asian': 1, 'African American': 2, 'Caucasian': 3, 'Hispanic': 4, 'Native American': 5},
                   "gender_mapping": {"Female": 1, "Male": 2}}


class Subgroup(Splitter):
    def __init__(self, feature_name, feature_value):
        self.feature_name = feature_name
        self.feature_value = feature_mapping[feature_name][feature_value]

    def split(self, dataset):
        all_indices = np.array(list(dataset.data['patientunitstayid'].unique()))

        train_indices = np.isin(all_indices, dataset.raw_data.loc[dataset.raw_data[self.feature_name] == self.feature_value, "patientunitstayid"])
        train_indices = np.where(train_indices)[0]
        train_indices = np.random.choice(train_indices, size=int(len(train_indices) * 0.8), replace=False)

        test_indices = np.setdiff1d(np.arange(len(all_indices)), train_indices)

        return [[train_indices, test_indices]]


splitters.register_builder("subgroup", Subgroup)
