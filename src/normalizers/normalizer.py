import abc


class Normalizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, x):
        raise NotImplementedError