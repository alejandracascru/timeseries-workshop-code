from sklearn.ensemble import RandomForestRegressor as SklearnRFRegressor

from .creation import models


class RFForecaster(SklearnRFRegressor):
    def __new__(cls, num_features=1, num_outputs=1, **kwargs):
        model = SklearnRFRegressor(**kwargs)

        return model


models.register_builder("rf_forecaster", RFForecaster)
