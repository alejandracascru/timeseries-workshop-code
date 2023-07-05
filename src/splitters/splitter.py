import abc


class Splitter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def split(self, dataset):
        raise NotImplementedError
