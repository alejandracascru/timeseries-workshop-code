import abc


class Padder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def pad(self, data):
        pass

    def df_to_list(self, data):
        grouped_df = data.groupby('patientunitstayid')
        df_list = []

        for idx, frame in grouped_df:
            df_list.append(frame)

        return df_list

