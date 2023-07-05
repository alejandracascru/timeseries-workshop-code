import numpy as np

from ..creation import feeders
from ..feeder import Feeder


class Sklearn(Feeder):
    def __init__(self, timesteps, **kwargs):
        self.timesteps = timesteps

    def feed(self, data, offset=0):
        x = []
        y = []

        for i in range(data.shape[1] - 1 - self.timesteps):
            sub_x = data[:, offset + i: self.timesteps + offset + i, 1:].reshape(-1)
            sub_y = data[:, self.timesteps + offset + i, -1].reshape(-1)

            x.append(sub_x)
            y.append(sub_y)

        x = np.stack(x)
        y = np.array(y)

        return x, y

    def feed_test(self, data, offset=0):
        x = data[0, offset: self.timesteps + offset, 1:].reshape(-1)
        y = data[0, self.timesteps + offset, -1].reshape(-1)

        return x.reshape((1, -1)), y

    def update(self, data, pred, i):
        data[:, self.timesteps + i, -1] = pred

    def future_steps(self, data, n):
        return data[:, : self.timesteps + n, -1].reshape(-1)


feeders.register_builder("sklearn_forecasting", Sklearn)
