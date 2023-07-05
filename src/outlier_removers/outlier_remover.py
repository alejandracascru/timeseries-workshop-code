import abc


class OutlierRemover(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def remove(self, data):
        raise NotImplementedError

