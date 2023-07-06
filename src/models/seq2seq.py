import torch

from .creation import models


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoding_size, feature_size, window_len, num_layer=2,
                 hidden_size=64, bidirectional=False):
        super().__init__()
        self.window_len = window_len
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.encoder_rnn = torch.nn.GRU(input_size=feature_size, hidden_size=hidden_size,
                                        num_layers=num_layer, batch_first=True,
                                        bidirectional=bidirectional)
        self.encoder_fc = torch.nn.Linear(
            in_features=hidden_size * num_layer * (2 if bidirectional else 1),
            out_features=encoding_size
        )
        self.decoder_rnn = torch.nn.GRUCell(input_size=feature_size, hidden_size=encoding_size)
        self.decoder_fc = torch.nn.Linear(
            in_features=encoding_size,
            out_features=feature_size
        )

    def encode(self, x):
        _, z = self.encoder_rnn(x)
        z = self.encoder_fc(z.reshape(len(x), -1))

        return z

    def decode(self, z):
        outputs = torch.zeros(self.window_len, len(z), self.feature_size)
        hidden = z  # torch.zeros(len(z), self.hidden_size)
        input = torch.zeros(len(z), self.feature_size)  # z

        for t in range(self.window_len):
            hidden = self.decoder_rnn(input, hidden)
            out = self.decoder_fc(hidden)
            outputs[t] = out
            input = out

        return outputs.permute((1, 0, 2))

    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)

        return reconstructed


models.register_builder("seq2seq", Seq2Seq)
