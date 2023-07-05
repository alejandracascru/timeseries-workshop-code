import numpy as np
from sklearn.model_selection import KFold as SklearnKFold

from .creation import splitters
from .splitter import Splitter


class Forecasting(Splitter):
    def __init__(self):
        self.kfold = SklearnKFold(n_splits=5, shuffle=True)

    def split(self, dataset):
        train_indices = np.array([0])
        test_indices = np.array([1])

        return [[train_indices, test_indices]]


splitters.register_builder("forecasting", Forecasting)
