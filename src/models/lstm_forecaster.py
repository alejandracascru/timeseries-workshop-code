import torch

from .creation import models


class LSTMForecaster(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_outputs, **kwargs):
        super(LSTMForecaster, self).__init__()

        self.lstm = torch.nn.LSTM(num_features, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)

        return self.fc(lstm_out[:, -1]).view(-1)



models.register_builder("lstm_forecaster", LSTMForecaster)
