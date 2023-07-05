import torch.nn

from .creation import modules
from .module import Module


class Classification(Module):
    def __init__(self, model, optimizer_args):
        super(Classification, self).__init__(model, optimizer_args)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        if len(out.shape) > 2:
            loss = self.criterion(torch.flatten(out, 0, 1), y.view(-1))
        else:
            loss = self.criterion(out, y)

        self.log("loss", loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        if len(out.shape) > 2:
            loss = self.criterion(torch.flatten(out, 0, 1), y.view(-1))
        else:
            loss = self.criterion(out, y)

        self.log("loss", loss, prog_bar=True, on_step=True)

        return loss


modules.register_builder("classification", Classification)
