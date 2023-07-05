import abc


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, model, data, feeder):
        raise NotImplementedError