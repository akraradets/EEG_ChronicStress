import pandas as pd
import numpy as np
import mne
import pickle 
import os

def clear_cache():
    os.remove("cache/PSS.pickle")


def save(data, filename):
    with open(f'cache/{filename}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(f'cache/{filename}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

def marker_converter_1(pandasRaw):
    marker_idx = list(pandasRaw['Marker'].unique())
    new_marker = []
    for data in pandasRaw['Marker']:
        if(data == '0' or data == 0):
            i = 0
        else:
            i = marker_idx.index(data)
        new_marker.append(i)
    set(new_marker)
    pandasRaw['Marker'] = new_marker
    pandasRaw = pandasRaw.drop(columns='timestamps')
    return pandasRaw,marker_idx

def marker_converter(pd_raw,marker_names):
    new_marker = []
    for data in pd_raw['Marker']:
        if(data == '0' or data == 0):
            i = 0
        else:
            i = marker_names.index(data)
        new_marker.append(i)
    pd_raw['Marker'] = new_marker
    pd_raw = pd_raw.drop(columns='timestamps')
    return pd_raw

def dataframe_to_raw(dataframe, sfreq):
    ch_names = list(dataframe.columns)
    ch_types = ['eeg'] * (len(dataframe.columns) - 1) + ['stim']
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    dataframe = dataframe.T  #mne looks at the tranpose() format
    dataframe[:-1] *= 1e-6  #convert from uVolts to Volts (mne assumes Volts data)
    # dataframe[:-1] *= 1e3  #convert from uVolts to Volts (mne assumes Volts data)

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq, verbose=False)

    raw = mne.io.RawArray(dataframe, info,verbose=False)
    raw.set_montage(ten_twenty_montage)
    # raw.plot()
    return raw

def get_section_from_marker(dataframe, start_marker:str, last_marker:str):
    if(len(start_marker.split('_')) != 3):
        raise ValueError(f"The input start_marker:{start_marker} is not in the correct format of [cat]_[block]_[no]")
    if(len(last_marker.split('_')) != 3):
        raise ValueError(f"The input last_marker:{last_marker} is not in the correct format of [cat]_[block]_[no]")
    
    start_index = dataframe.loc[dataframe['Marker'] == start_marker].index
    # Check that there is one and only index
    if(len(start_index) != 1):
        raise ValueError(f"The number of index of marker:{start_marker} is not 1\nindex:{start_index}")

    last_index = dataframe.loc[dataframe['Marker'] == last_marker].index
    # Check that there is one and only index
    if(len(last_index) != 1):
        raise ValueError(f"The number of index of marker:{last_marker} is not 1\nindex:{last_index}")

    start_index = start_index[0]
    last_index = last_index[0]

    break_marker = '-1'
    break_indexes = dataframe.loc[dataframe['Marker'] == break_marker].index
    break_index =  break_indexes[np.argmin(abs(break_indexes - last_index))]
    if(break_index < last_index):
        break_index = last_index + 250
    section = dataframe.iloc[start_index:break_index].copy()
    section.reset_index(drop=True, inplace=True)
    return section

def get_section_from_catblock(dataframe, category:int, block:int):
    if(type(category) != int):
        raise ValueError(f"category must be type {int}")
    if(category not in [1,2,3,4,5]):
        raise ValueError(f"category must be in range {[1,2,3,4,5]}")
    
    if(type(block) != int):
        raise ValueError(f"block must be type {int}")
    if(block not in [1,2]):
        raise ValueError(f"block must be in range {[1,2]}")
    

    start_marker = f"{category}_{block}_1"
    last_marker = f"{category}_{block}_5"
    return get_section_from_marker(dataframe, start_marker, last_marker)