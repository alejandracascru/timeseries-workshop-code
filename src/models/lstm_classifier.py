import torch

from .creation import models


class LSTMClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_outputs):
        super(LSTMClassifier, self).__init__()

        self.lstm = torch.nn.LSTM(num_features, hidden_size, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * 2, num_outputs)


    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        return self.fc(lstm_out[:, -1])


models.register_builder("lstm_classifier", LSTMClassifier)
