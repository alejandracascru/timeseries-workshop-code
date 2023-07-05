import numpy as np


from .creation import padders
from .padder import Padder

from src.data.padded_array import PaddedArray


class Basic(Padder):
    def __init__(self, max_len):
        self.max_len = max_len

    def pad(self, data):
        df_list = self.df_to_list(data)

        padded_data = []
        observations_per_patient = []
        patient_ids = []

        for item in df_list:
            tmp = np.zeros((self.max_len, item.shape[1])).astype("float32")
            tmp[:min(self.max_len, item.shape[0]), :item.shape[1]] = item[:min(self.max_len, item.shape[0])]
            padded_data.append(tmp)
            observations_per_patient.append(min(self.max_len, item.shape[0]))
            patient_ids.append(item["patientunitstayid"].iloc[0])

        padded_data = PaddedArray(padded_data, observations_per_patient, patient_ids)

        return padded_data


padders.register_builder("basic", Basic)