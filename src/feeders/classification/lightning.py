import torch.utils.data

from src.data.sequence_dataset import SequenceDataset
from ..creation import feeders
from ..feeder import Feeder


class Lightning(Feeder):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def feed(self, data):
        x, y = data[:, :, 1:-1], data[:, 0, -1:]

        return x, y

    def train_dataloader(self, data):
        x, y = data[:, :, 1:-1], data[:, 0, -1:].reshape(-1).astype("long")

        dataset = SequenceDataset(x, y)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self, data):
        x, y = data[:, :, 1:-1], data[:, 0, -1:].reshape(-1).astype("long")

        dataset = SequenceDataset(x, y)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


feeders.register_builder("lightning_classification", Lightning)
