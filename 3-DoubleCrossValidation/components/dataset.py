from __future__ import annotations
import os
import pandas as pd
import numpy as np
import mne
class Dataset:
    LABEL_NON_STRESSED = 0
    LABEL_NEUTRAL = 2
    LABEL_STRESSED = 1

    def __init__(self, dataset_path:str, EEG_folder:str="EEG/", PSS_name:str="PSS.csv"): 
        """
            dataset_path: path to the dataset. Inside must have have `EEG` as a folder and `PSS.csv` file.
            EEG_folder: if you want to change the default `EEG` folder to something else.
            PSS_name: if you want to change the default `PSS.csv` filename to something else.
        """
        #### Assert the files ####
        EEG_path = f"{dataset_path}/{EEG_folder}"
        PSS_path = f"{dataset_path}/{PSS_name}"
        assert os.path.exists(EEG_path), f"{EEG_path} does not exist."
        assert os.path.exists(PSS_path), f"{PSS_path} does not exist."

        #### Load PSS files ####
        PSS = pd.read_csv(PSS_path, index_col="No.", )

        #### Check EEG records ####
        self.names = []
        self.files = []
        self.attrs = []
        self.scores = []
        for index in PSS.index:
            name = f"{index:03d}"
            self.names.append(f"{name}")
            file = f"{EEG_path}{name}.csv"
            assert os.path.exists(file), f"{file} is not exist."
            assert file not in self.files, f"{name} is duplicated."
            self.files.append(file)
            attr = {
                'Gender': PSS.loc[index, 'Gender'],
                'MBTI': PSS.loc[index, 'MBTI'],
                'Age': PSS.loc[index, 'Age'],
            }
            self.attrs.append(attr)
            self.scores.append(PSS.loc[index, 'PSS Score'])

        print(f"Found: {len(self.files)} files")

        #### Init Attribute ####
        self.data = []
        self.labels = self._calculate_label()

        print(f"Non-stressed:{sum(self.labels == self.LABEL_NON_STRESSED)}")
        print(f"Stressed:{sum(self.labels == self.LABEL_STRESSED)}")
        print(f"Neutral:{sum(self.labels == self.LABEL_NEUTRAL)}")

    def set_sampling_rate(self, sampling_rate:int):
        self.sampling_rate = sampling_rate

    def set_marker(self, start_minute:float, stop_minute:float, segment:int):
        sampling_rate = self.sampling_rate #Hz
        step_minute = segment/60
        # 15/60 = 0.25
        steps = np.arange(start_minute,stop_minute,step_minute)
        steps = np.expand_dims(steps * sampling_rate * 60, axis=1)
        marker = np.concatenate( [steps, np.zeros( steps.shape ), np.ones( steps.shape ) ], axis=1  ).astype(np.int64)
        self.marker = marker
        print(f"marker.shape={self.marker.shape}")

    def _calculate_label(self) -> np.ndarray:
        N = len(self.scores)
        mu = sum(self.scores)/N
        std = (sum((np.array(self.scores) - mu)**2)/N)**0.5
        print(f"Mean:{mu}, Std:{std}")
        Tu = mu + (std/2)
        Tl = mu - (std/2)

        labels = []
        for score in self.scores:
            if(score < Tl): labels.append(self.LABEL_NON_STRESSED)
            elif(Tl <= score <= Tu): labels.append(self.LABEL_NEUTRAL)
            elif(score > Tu): labels.append(self.LABEL_STRESSED)
        return np.array(labels)

    def get_index(self, name:str) -> int:
        """
            name: the index of files
        """
        assert name in self.names, f"name={name} does nto exists."
        return self.names.index(name)

    def load_data(self, name:str):
        """
            name: the index of files
        """
        index = self.get_index(name)
        df = pd.read_csv(self.files[index], index_col='Index')
        epochs = self._convert_df_to_mne_raw(df=df)

    def _convert_df_to_mne_raw(self, df:pd.DataFrame) -> mne.io.RawArray:
        # convert data to mne.Epochs
        # ch_names = list(df.columns[1:-1])
        ch_names = ['Fp1','Fp2','F3','F4','F7','F8','C3','C4','T3','T4','T5','T6','P3','P4','O1','O2']
        ch_types = ['eeg'] * len(ch_names)
        # https://mne.tools/stable/generated/mne.create_info.html
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=self.sampling_rate) # type: ignore

        df = df.T  #mne looks at the tranpose() format
        df[:-1] *= 1e-6  #convert from uVolts to Volts (mne assumes Volts data)

        raw = mne.io.RawArray(df,info=info,verbose='CRITICAL')
        raw.set_montage('standard_1020')
        return raw
    

class Dataset_Builder:
    def __init__(self, dataset_path:str, EEG_folder:str="EEG/", PSS_name:str="PSS.csv"):
        self.is_sampling_rate_set = False
        self.is_marker_set = False
        self.dataset = Dataset(dataset_path=dataset_path, EEG_folder=EEG_folder, PSS_name=PSS_name)
    
    def with_sampling_rate(self, sampling_rate:int=125) -> Dataset_Builder:
        self.dataset.set_sampling_rate(sampling_rate)
        self.is_sampling_rate_set = True
        return self

    def with_marker(self, start_minute:float=0.0, stop_minute:float=5.0, segment:int=15) -> Dataset_Builder:
        self.dataset.set_marker(start_minute, stop_minute, segment)
        self.is_marker_set = True
        return self

    def build(self) -> Dataset:
        assert self.is_sampling_rate_set, f"You have not yet set the `sampling_rate`."
        assert self.is_marker_set, f"You have not yet set the `marker`." 
        return self.dataset

if __name__ == "__main__":
    dataset_path = "3-DoubleCrossValidation/data"
    dataset = Dataset_Builder(dataset_path=dataset_path)\
                .with_sampling_rate(125)\
                .with_marker(start_minute=1, stop_minute=3, segment=6)\
                .build()
    # dataset = Dataset(dataset_path=dataset_path)
    dataset.load_data('001')

