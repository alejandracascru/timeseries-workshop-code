import torch

from .creation import models


class GRUClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_outputs):
        super(GRUClassifier, self).__init__()

        self.gru = torch.nn.GRU(num_features, hidden_size, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * 2, num_outputs)

    def forward(self, x):
        gru_out, _ = self.gru(x)

        return self.fc(gru_out[:, -1])


models.register_builder("gru_classifier", GRUClassifier)
