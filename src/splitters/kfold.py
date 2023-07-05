import numpy as np
from sklearn.model_selection import KFold as SklearnKFold

from .creation import splitters
from .splitter import Splitter


class KFold(Splitter):
    def __init__(self):
        self.kfold = SklearnKFold(n_splits=5, shuffle=True)

    def split(self, dataset):
        all_indices = np.array(list(dataset.data['patientunitstayid'].unique()))

        return self.kfold.split(all_indices)


splitters.register_builder("kfold", KFold)
