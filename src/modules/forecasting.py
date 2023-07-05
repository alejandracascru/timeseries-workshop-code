import torch.nn

from .creation import modules
from .module import Module


class Forecasting(Module):
    def __init__(self, model, optimizer_args):
        super(Forecasting, self).__init__(model, optimizer_args)

        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        loss = self.criterion(out, y)

        self.log("loss", loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        loss = self.criterion(out, y)

        self.log("loss", loss, prog_bar=True, on_step=True)

        return loss


modules.register_builder("forecasting", Forecasting)
