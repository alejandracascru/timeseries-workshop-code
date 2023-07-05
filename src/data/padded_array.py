import numpy as np


class PaddedArray(np.ndarray):
    def __new__(cls, input_list, observations_per_patient, patients_ids, *args, **kwargs):
        obj = np.asarray(input_list).view(cls)

        obj._observations_per_patient = observations_per_patient
        obj._patient_ids = patients_ids

        return obj

    @property
    def observations_per_patient(self):
        return self._observations_per_patient

    @property
    def patient_ids(self):
        return self._patient_ids
