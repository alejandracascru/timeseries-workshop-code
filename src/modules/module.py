from pytorch_lightning import LightningModule

from src.optimizers import optimizers


class Module(LightningModule):
    def __init__(self, model, optimizer_args):
        super(Module, self).__init__()

        self.model = model
        self.optimizer_args = optimizer_args

    def configure_optimizers(self):
        optimizer = optimizers.create(self.optimizer_args.name, parameters=self.model.parameters(),
                                      **self.optimizer_args.params)

        return [optimizer]
