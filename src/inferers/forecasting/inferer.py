import abc


class Inferer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def one_step_forecast(self, model, data, feeder):
        raise NotImplementedError

    @abc.abstractmethod
    def n_step_forecast(self, model, data, feeder, n):
        raise NotImplementedError