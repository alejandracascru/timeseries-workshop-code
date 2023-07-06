from sklearn.ensemble import RandomForestClassifier as SklearnRFClassifier

from .creation import models


class RFClassifier(SklearnRFClassifier):
    def __new__(cls, num_features=1, num_outputs=1, **kwargs):
        model = SklearnRFClassifier(**kwargs)

        return model


models.register_builder("rf_classifier", RFClassifier)
