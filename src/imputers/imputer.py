import abc


class Imputer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def impute(self, data):
        raise NotImplementedError