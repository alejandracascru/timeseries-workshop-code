from .outlier_remover import OutlierRemover
from .creation import outlier_removers


class Identity(OutlierRemover):
    def remove(self, data):
        return data


outlier_removers.register_builder("identity", Identity)