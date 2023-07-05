import copy

import torch
from pytorch_lightning import LightningModule, Trainer

from .inferer import Inferer
from ..creation import inferers


class Lightning(Inferer):
    def one_step_forecast(self, model, data, feeder):
        module = Module(model)
        out = Trainer(use_distributed_sampler=False, devices=1).predict(module, feeder.test_dataloader(data))

        out = torch.cat(out)

        return out.numpy()

    def n_step_forecast(self, model, data, feeder, n):
        data = copy.deepcopy(data)
        module = Module(model)
        trainer = Trainer(use_distributed_sampler=False, devices=1)
        preds = []

        for i in range(n):
            out = trainer.predict(module, feeder.test_dataloader(data, i))
            out = torch.cat(out)
            preds.append(out.view(-1))

            feeder.update(data, out, i)

        preds = torch.cat(preds)

        return preds.numpy()


class Module(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx):
        x, y = batch

        out = self.model(x)

        return out


inferers.register_builder("lightning_forecasting", Lightning)
