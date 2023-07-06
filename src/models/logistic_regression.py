from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from .creation import models


class LogisticRegression(SklearnLogisticRegression):
    def __new__(cls, num_features=1, num_outputs=1, **kwargs):
        model = SklearnLogisticRegression(**kwargs)

        return model


models.register_builder("logistic_regression", LogisticRegression)
