import torch.utils.data

from src.data.forecasting_dataset import ForecastingDataset
from ..creation import feeders
from ..feeder import Feeder


class Lightning(Feeder):
    def __init__(self, batch_size, timesteps, **kwargs):
        self.batch_size = batch_size
        self.timesteps = timesteps

    def feed(self, data, offset=0):
        x = data[:, offset:self.timesteps + offset, 1:]
        y = data[:, self.timesteps + offset, -1]

        return x, y

    def train_dataloader(self, data, offset=0):
        x = data[:, :, 1:]

        dataset = ForecastingDataset(x, self.timesteps, True, offset)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self, data, offset=0):
        x = data[:, :, 1:]

        dataset = ForecastingDataset(x, self.timesteps, False, offset)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def update(self, data, pred, i):
        data[:, self.timesteps + i, -1] = pred

    def future_steps(self, data, n):
        return data[:, : self.timesteps + n, -1].reshape(-1)


feeders.register_builder("lightning_forecasting", Lightning)
