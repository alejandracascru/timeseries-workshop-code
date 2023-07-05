import numpy as np


class EICU:
    def __init__(self, path, use_cat, use_num):
        self.path = path
        self.data = None

        self.data_dimension = None

        self.use_cat = use_cat
        self.use_num = use_num

    @property
    def num_features(self):
        return self.data.shape[1] - 2  # -2 to get rid of index and label

    def set_stats(self, x, y):
        self.data_dimension = x.shape[1]
        self.num_classes = len(np.unique(y))

    def _prepare_categorical_columns(self, data):
        columns_ord = ["patientunitstayid", "itemoffset",
                       "Eyes", "Motor", "GCS Total", "Verbal",
                       "ethnicity", "gender", "apacheadmissiondx",
                       "FiO2", "Heart Rate", "Invasive BP Diastolic",
                       "Invasive BP Systolic", "MAP (mmHg)", "O2 Saturation",
                       "Respiratory Rate", "Temperature (C)", "admissionheight",
                       "admissionweight", "age", "glucose", "pH",
                       "hospitaladmitoffset",
                       "hospitaldischargestatus", "unitdischargeoffset",
                       "unitdischargestatus"]
        data = data[data.gender != 0]  # unknown gender is dropped
        data = data[data.hospitaldischargestatus != 2]  # unknown hospital discharge is dropped
        data = data[columns_ord]

        data.apacheadmissiondx = data.apacheadmissiondx.astype(int, errors="ignore")
        data.ethnicity = data.ethnicity.astype(int, errors="ignore")
        data.gender = data.gender.astype(int, errors="ignore")
        data["GCS Total"] = data["GCS Total"].astype(int, errors="ignore")
        data["Eyes"] = data["Eyes"].astype(int, errors="ignore")
        data["Motor"] = data["Motor"].astype(int, errors="ignore")
        data["Verbal"] = data["Verbal"].astype(int, errors="ignore")
        data.apacheadmissiondx = data.apacheadmissiondx + 1
        data.ethnicity = data.ethnicity + 1
        dxmax = data.apacheadmissiondx.max()

        data["GCS Total"] = data["GCS Total"] + dxmax

        return data
