import torch
from pytorch_lightning import LightningModule, Trainer

from .inferer import Inferer
from ..creation import inferers


class Lightning(Inferer):
    def predict(self, model, data, feeder):
        probs = self.predict_proba(model, data, feeder)
        preds = (probs > 0.5).astype(int)

        return preds

    def predict_proba(self, model, data, feeder):
        module = Module(model)
        probs =  Trainer(use_distributed_sampler=False, devices=1).predict(module, feeder.test_dataloader(data))

        probs = torch.cat(probs)

        if len(probs.shape) > 2:
            probs = probs[:, :, 1]
        else:
            probs = probs[:, 1]

        return probs.numpy()


class Module(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx):
        x, y = batch

        out = self.model(x)
        probs = torch.nn.functional.softmax(out, dim=1)

        return probs


inferers.register_builder("lightning_classification", Lightning)
