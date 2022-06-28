from mne_features.feature_extraction import FeatureExtractor, extract_features
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
from scipy.integrate import simps
import numpy as np
import logging
import pprint
import os

def preprocessing(data:np.ndarray, sampling_rate:int, bands:dict, log10:bool) -> np.ndarray:
    logger = logging.getLogger('main')
    data = PSD(data=data, sampling_rate=sampling_rate, bands=bands)
    data = relative(data=data)
    if(log10):
        data = 10 * np.log10(data)
    assert np.isnan(data).any() == False
    data = asymetries(data=data)
    # Upon testing, this StandardScaler will apply (X_i - Mu_i) / std_i for each i (column/feature) individually
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    assert np.isnan(data).any() == False
    return data


def PSD(data:np.ndarray, sampling_rate:int, bands:dict) -> np.ndarray:
    logger = logging.getLogger('main')
    # https://raphaelvallat.com/bandpower.html

    # Compute the modified periodogram (Welch)
    freqs, psds = welch(x=data, fs=sampling_rate, nperseg=sampling_rate * 2) # psd has a unit of V^2/Hz
    # logger.info(f"{freqs.shape=}|{psds.shape=}")
    temp = []
    for name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = simps(psds[:,:,idx_band], dx=freqs[1] - freqs[0]) # band_power has a unit of V^2
        # logger.info(f"{band_power.shape=}")
        temp.append(band_power)

    temp = np.hstack(temp)
    assert (temp == abs(temp)).all()
    assert np.isnan(temp).any() == False
    assert temp.shape[1] == (16 * 7), f"The second dimension should be (16 * 7). {temp.shape=}"
    return temp


def PSD_old(data:np.ndarray,  sampling_rate:int) -> np.ndarray:
    logger = logging.getLogger('main')
    # [alias_feature_function]__[optional_param]
    params = {
        'pow_freq_bands__log':False,
        'pow_freq_bands__normalize':False,
        'pow_freq_bands__psd_method':'welch',
        'pow_freq_bands__freq_bands':{
            'delta': [1,3],
            'theta': [4,7],
            'alpha': [8,12],
            'beta': [13,30],
            'gamma': [25,43],
            'slow': [4,13],
            'beta_low': [13,17]
        } 
    }
    logger.info(f"""extractor_params={pprint.pformat(params, indent=2)}""")

    fe = FeatureExtractor(sfreq=sampling_rate, selected_funcs=['pow_freq_bands'],params=params,n_jobs=os.cpu_count())
    data = fe.fit_transform(X=data)
    assert data.shape[1] == (16 * 7), f"The second dimension should be (16 * 7). {data.shape=}"

    return data

def relative(data:np.ndarray) -> np.ndarray:
    logger = logging.getLogger('main')
    # Here the data shape changes to (n_sample, n_channels * 7)
    # In the code of mne_features, they use psd.ravel() which yield a flattern version of the array
    # the default method of this is ravel('C') -> https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
    # Thus, it cancatenates from [row_0, row_1, row_2, ...]
    # Here row_0 means channel_1 and so on
    # Each row consits of 7 freq bands
    # Now we are going to unravel this.
    # After I di some experiment, the reshape works.

    ## Calculate relative ##
    index_gamma = _get_index(band_name='gamma')
    index_slow = _get_index(band_name='slow')
    gammas = data[:, index_gamma * 16 : (index_gamma+1) * 16 ] 
    slows = data[:, index_slow * 16 : (index_slow+1) * 16 ]
    # relative = ratio of gamma to slow = gamma / slow
    relatives = gammas / slows
    data = np.hstack([data, relatives])
    assert np.isnan(data).any() == False
    assert data.shape[1] == (16 * 8), f"The second dimension should be (16 * 8). {data.shape=}"
    return data

def asymetries(data:np.ndarray) -> np.ndarray:
    logger = logging.getLogger('main')
    F4_alpha = data[:, _get_index('F4','alpha')]
    F3_alpha = data[:, _get_index('F3','alpha')]
    T4_alpha = data[:, _get_index('T4','alpha')]
    T3_alpha = data[:, _get_index('T3','alpha')]
    alpha_f = (F4_alpha - F3_alpha) / (F4_alpha + F3_alpha)
    alpha_t = (T4_alpha - T3_alpha) / (T4_alpha + T3_alpha)
    alpha_a = alpha_f + alpha_t

    F4_beta = data[:, _get_index('F4','beta')]
    F3_beta = data[:, _get_index('F3','beta')]
    T4_beta = data[:, _get_index('T4','beta')]
    T3_beta = data[:, _get_index('T3','beta')]
    beta_f = (F4_beta - F3_beta) / (F4_beta + F3_beta)
    beta_t = (T4_beta - T3_beta) / (T4_beta + T3_beta)

    alpha_f = np.expand_dims(alpha_f, axis=1)
    alpha_t = np.expand_dims(alpha_t, axis=1)
    alpha_a = np.expand_dims(alpha_a, axis=1)
    beta_f = np.expand_dims(beta_f, axis=1)
    beta_t = np.expand_dims(beta_t, axis=1)

    # 25-06-2022 18:24:51|preprocessing.py:73|INFO|data.shape=(420, 128), alpha_f.shape=(420,), alpha_t.shape=(420,), alpha_a.shape=(420,), beta_f.shape=(420,), beta_t.shape=(420,)
    logger.info(f"{data.shape=}, {alpha_f.shape=}, {alpha_t.shape=}, {alpha_a.shape=}, {beta_f.shape=}, {beta_t.shape=}")
    data = np.hstack([data, alpha_f, alpha_t, alpha_a, beta_f, beta_t])
    assert np.isnan(data).any() == False
    assert data.shape[1] == (16 * 8) + 5, f"The second dimension should be (16*8)+5=133. {data.shape=}"
    return data

def _get_index(ch_name:str = None, band_name:str = None) -> int: # type : ignore
    logger = logging.getLogger('main')
    ch_names = ['Fp1','Fp2','F3','F4','F7','F8','C3','C4','T3','T4','T5','T6','P3','P4','O1','O2']
    band_names = ['delta','theta','alpha','beta','gamma','slow','beta_low']
    assert ((ch_name == None) and (band_name == None)) == False, f"Both ch_name and band_name can not be None at the same time"
    if(band_name == None):
        return ch_names.index(ch_name)
    elif(ch_name == None):
        return band_names.index(band_name)
    else:
        return ch_names.index(ch_name) + (band_names.index(band_name) * 16)