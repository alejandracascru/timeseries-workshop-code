from .inferer import Inferer

from ..creation import inferers


class Sklearn(Inferer):
    def predict(self, model, data, feeder):
        x, _ = feeder.feed(data)
        return model.predict(x)

    def predict_proba(self, model, data, feeder):
        x, _ = feeder.feed(data)
        probs = model.predict_proba(x)

        if probs.shape[1] == 2:
            return probs[:, 1]

        return probs


inferers.register_builder("sklearn_classification", Sklearn)
