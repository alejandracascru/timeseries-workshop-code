import numpy as np
from sklearn.model_selection import train_test_split

from .creation import splitters
from .splitter import Splitter


class TrainTestSplit(Splitter):
    def split(self, dataset):
        all_indices = np.arange(len(dataset.data['patientunitstayid'].unique()))

        return [train_test_split(all_indices, test_size=0.2)]


splitters.register_builder("train_test_split", TrainTestSplit)
