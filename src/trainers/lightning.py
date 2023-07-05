from .creation import trainers
from .trainer import Trainer
from src.modules import modules
from pytorch_lightning import Trainer as LightningTrainer


class Lightning(Trainer):
    def __init__(self, module_args, *args, **kwargs):
        super(Lightning, self).__init__()

        self.module_args = module_args
        self.trainer = LightningTrainer(*args, **kwargs)

    def fit(self, model, data, feeder):
        module = modules.create(self.module_args.name, model=model, **self.module_args.params)

        self.trainer.fit(module, feeder.train_dataloader(data))


trainers.register_builder("lightning", Lightning)



