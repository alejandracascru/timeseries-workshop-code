from .trainer import Trainer
from .creation import trainers


class Sklearn(Trainer):
    def fit(self, model, data, feeder):
        x, y = feeder.feed(data)
        model.fit(x, y)


trainers.register_builder("sklearn", Sklearn)