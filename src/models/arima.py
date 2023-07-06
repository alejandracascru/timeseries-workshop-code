from statsmodels.tsa.arima.model import ARIMA as StatsmodelARIMA

from .creation import models


class ARIMA:
    def __init__(self, num_features, num_outputs, p, d, q):
        self.p = p
        self.d = d
        self.q = q

        self.model = None
        self.data_len = None

    def fit(self, x):
        self.model = StatsmodelARIMA(x, order=(self.p, self.d, self.q))
        self.model = self.model.fit()

        self.data_len = len(x)

    def predict(self, n):
        return self.model.forecast(n).reshape((1, -1))


models.register_builder("arima", ARIMA)
