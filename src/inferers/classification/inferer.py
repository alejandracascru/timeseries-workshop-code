import abc


class Inferer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, model, data, feeder):
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, model, data, feeder):
        raise NotImplementedError
