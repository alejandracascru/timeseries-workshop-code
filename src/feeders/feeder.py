import abc


class Feeder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feed(self, data):
        raise NotImplementedError
