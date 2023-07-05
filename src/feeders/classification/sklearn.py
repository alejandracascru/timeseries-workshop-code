from ..creation import feeders
from ..feeder import Feeder


class Sklearn(Feeder):
    def feed(self, data):
        x, y = data[:, :, 1:-1], data[:, 0, -1:]
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))

        return x, y


feeders.register_builder("sklearn_classification", Sklearn)