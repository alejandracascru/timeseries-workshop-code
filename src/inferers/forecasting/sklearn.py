import copy

import numpy as np

from .inferer import Inferer
from ..creation import inferers


class Sklearn(Inferer):
    def one_step_forecast(self, model, data, feeder):
        x, _ = feeder.feed(data)
        return model.predict(x)

    def n_step_forecast(self, model, data, feeder, n):
        data = copy.deepcopy(data)
        preds = []

        for i in range(n):
            x, _ = feeder.feed_test(data, i)
            out = model.predict(x)
            preds.append(out.reshape(-1))

            feeder.update(data, out, i)

        preds = np.concatenate(preds)

        return preds


inferers.register_builder("sklearn_forecasting", Sklearn)
