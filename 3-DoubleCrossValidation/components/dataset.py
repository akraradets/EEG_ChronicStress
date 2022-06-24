from __future__ import annotations
import os
from tracemalloc import start
from typing import Tuple
import pandas as pd
import pickle
import numpy as np
import mne
import sklearn
class Dataset:
    GROUP_NON_STRESSED = 0
    GROUP_NEUTRAL = 2
    GROUP_STRESSED = 1

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
            file = f"{EEG_path}{name}.pickle"
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
        self.groups = self._calculate_group()

        print(f"Non-stressed:{sum(self.groups == self.GROUP_NON_STRESSED)}")
        print(f"Stressed:{sum(self.groups == self.GROUP_STRESSED)}")
        print(f"Neutral:{sum(self.groups == self.GROUP_NEUTRAL)}")

    def set_sampling_rate(self, sampling_rate:int):
        self.sampling_rate = sampling_rate

    def set_marker(self, start_minute:float, stop_minute:float, segment_second:int):
        """
            start_minute: the marker will start at this minute
            stop_minute: the marker will not exceed this minute
            segment: number of second of each segment
        """

        start_sample = start_minute * self.sampling_rate * 60
        stop_sample = stop_minute * self.sampling_rate * 60
        step_sample = segment_second * self.sampling_rate

        assert (stop_sample - start_sample) % step_sample == 0, f"The number segment_second={segment_second} is not valid. Please select the number that can divide stop_minute - start_minute."
        print(f"number of epochs = {(stop_sample - start_sample)/step_sample}")
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        self.step_sample = step_sample
        self._expected_epochs_number = int((stop_sample - start_sample)/step_sample)
        # steps = np.arange(start_sample,stop_sample,step_sample)
        # steps = np.expand_dims(steps, axis=1)
        # marker = np.concatenate( [steps, np.zeros( steps.shape ), np.ones( steps.shape ) ], axis=1  ).astype(np.int64)
        # self.marker = marker
        # print(f"marker.shape={self.marker.shape}")

    def _calculate_group(self) -> np.ndarray:
        N = len(self.scores)
        mu = sum(self.scores)/N
        std = (sum((np.array(self.scores) - mu)**2)/N)**0.5 # type: ignore
        print(f"Mean:{mu}, Std:{std}")
        Tu = mu + (std/2)
        Tl = mu - (std/2)

        groups = []
        for score in self.scores:
            if(score < Tl): groups.append(self.GROUP_NON_STRESSED)
            elif(Tl <= score <= Tu): groups.append(self.GROUP_NEUTRAL)
            elif(score > Tu): groups.append(self.GROUP_STRESSED)
        return np.array(groups)

    def _get_index(self, name:str) -> int:
        """
            name: the index of files
        """
        assert name in self.names, f"name={name} does nto exists."
        return self.names.index(name)

    def _load_pickle(self, pickle_path:str) -> np.ndarray:
        with open(pickle_path, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def _get_data(self, name:str) -> np.ndarray:
        index = self._get_index(name)
        data = self._load_pickle(self.files[index])
        return data

    def _get_group(self, name:str) -> int:
        index = self._get_index(name)
        group = self.groups[index]
        return group

    def load_data(self, name:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            name: the index of files
        """
        data = self._get_data(name)
        assert data.shape == (16,37500), f"data.shape={data.shape}"
        group = self._get_group(name)

        epochs = np.split(data[:, self.start_sample:self.stop_sample], self._expected_epochs_number, axis=1)
        # [np.array(16,125*segment),...] => self._expected_epochs_number
        epochs = [np.expand_dims(epochs,axis=0) for epochs in epochs]
        # [np.array(1,16,125*segment),...] => self._expected_epochs_number
        epochs = np.vstack(epochs)
        # (self._expected_epochs_number , 16 , 125*segment )
        assert epochs.shape[0] == self._expected_epochs_number,\
            f"epochs.shape={epochs.shape}. Expect the first dimension to be {self._expected_epochs_number}"

        labels = np.array([group] * epochs.shape[0])
        sklearn_groups = np.array([self._get_index(name)] * epochs.shape[0])

        return epochs, labels, sklearn_groups

    def load_data_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        epochs = []
        labels = []
        sklearn_groups = []

        for name, group in zip(self.names, self.groups):
            if(group == self.GROUP_NEUTRAL): continue
            a,b,c = self.load_data(name)
            epochs.append(a)
            labels.append(b)
            sklearn_groups.append(c)
        epochs = np.vstack(epochs)
        labels = np.hstack(labels)
        sklearn_groups = np.hstack(sklearn_groups)

        return epochs, labels, sklearn_groups 

    def _convert_df_to_mne_raw(self, df:pd.DataFrame) -> mne.io.RawArray:
        # convert data to mne.Epochs
        ch_names = list(df.columns[1:-1])
        # ch_names = ['Fp1','Fp2','F3','F4','F7','F8','C3','C4','T3','T4','T5','T6','P3','P4','O1','O2']
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

    def with_marker(self, start_minute:float=0.0, stop_minute:float=5.0, segment_second:int=15) -> Dataset_Builder:
        self.dataset.set_marker(start_minute, stop_minute, segment_second)
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
                .with_marker(start_minute=1, stop_minute=2, segment_second=5)\
                .build()

    data, labels, groups = dataset.load_data_all()
    data.shape, labels.shape, groups.shape

