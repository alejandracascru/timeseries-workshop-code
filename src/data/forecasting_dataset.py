import torch


class ForecastingDataset(torch.utils.data.Dataset):
    def __init__(self, x, timesteps, train, offset=0):
        self.x = x
        self.timesteps = timesteps
        self.offset = offset
        self.train = train


    def __len__(self):
        if self.train:
            return self.x.shape[0] * (self.x.shape[1] - self.timesteps - 1)
        else:
            return len(self.x)

    def __getitem__(self, idx):
        if self.train:
            chunk_size = self.x.shape[1] - self.timesteps
            sample_idx = idx // chunk_size
            time_idx = idx % chunk_size
        else:
            sample_idx = idx
            time_idx = 0

        x, y = self.x[sample_idx, self.offset + time_idx:self.offset + time_idx +  self.timesteps].reshape(self.timesteps, self.x.shape[2]), self.x[
            sample_idx, self.offset + time_idx + self.timesteps, -1]

        return x, y
