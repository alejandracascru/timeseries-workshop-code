from hyperimpute.plugins.imputers import Imputers

from .creation import imputers
from .imputer import Imputer


class HyperImpute(Imputer):
    def impute(self, data):
        imp = Imputers().get('hyperimpute', optimizer="hyperband")

        data = data.groupby(['patientunitstayid']).apply(
            lambda group: imp.fit_transform(group.copy())
        )
        return data


imputers.register_builder("hyperimpute", HyperImpute)
