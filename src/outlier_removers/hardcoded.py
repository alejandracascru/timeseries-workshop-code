import copy

from .outlier_remover import OutlierRemover
from .creation import outlier_removers


class Hardcoded(OutlierRemover):
    def __init__(self):
        self.value_ranges = {"Eyes": [0, 5],
                             "GCS Total": [2, 16],
                             "Heart Rate": [0, 350],
                             "Motor": [0, 6],
                             "Invasive BP Diastolic": [0, 375],
                             "Invasive BP Systolic": [0, 375],
                             "MAP (mmHg)": [14, 330],
                             "Verbal": [1, 5],
                             "admissionheight": [100, 240],
                             "admissionweight": [30, 250],
                             "glucose": [33, 1200],
                             "pH": [6.3, 10],
                             "FiO2": [15, 110],
                             "O2 Saturation": [0, 100],
                             "Respiratory Rate": [0, 100],
                             "Temperature (C)": [26, 45]}

    def remove(self, data):
        data = copy.deepcopy(data)

        # clip all values to
        for key, value_range in self.value_ranges.items():
            if key in data.columns:
                data[key] = data[key].clip(value_range[0], value_range[1])

        return data


outlier_removers.register_builder("hardcoded", Hardcoded)