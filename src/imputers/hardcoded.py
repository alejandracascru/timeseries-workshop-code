from .creation import imputers
from .imputer import Imputer


class Hardcoded(Imputer):
    def __init__(self):
        self.normal_values = {'Eyes': 4, 'GCS Total': 15, 'Heart Rate': 86, 'Motor': 6, 'Invasive BP Diastolic': 56,
                              'Invasive BP Systolic': 118, 'O2 Saturation': 98, 'Respiratory Rate': 19,
                              'Verbal': 5, 'glucose': 128, 'admissionweight': 81, 'Temperature (C)': 36,
                              'admissionheight': 170, "MAP (mmHg)": 77, "pH": 7.4, "FiO2": 0.21}

    def impute(self, data):
        return data.fillna(value=self.normal_values)


imputers.register_builder("hardcoded", Hardcoded)
