from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from .creation import models


class LinearRegression(SklearnLinearRegression):
    def __new__(cls, num_features=1, num_outputs=1, **kwargs):
        model = SklearnLinearRegression(**kwargs)

        return model


models.register_builder("linear_regression", LinearRegression)
