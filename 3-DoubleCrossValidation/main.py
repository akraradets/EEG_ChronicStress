from components.dataset import Dataset_Builder
from components.preprocessing import preprocessing
from components.ml import train_model
from components.logger import init_logger
from components.stats import check_normality, rank_mannwhitneyu
from components.utils import *
import numpy as np
import logging 
import time

def get_feature_name():
    import itertools
    ch_names = ['Fp1','Fp2','F3','F4','F7','F8','C3','C4','T3','T4','T5','T6','P3','P4','O1','O2']
    band_names = ['delta','theta','alpha','beta','gamma','slow','beta_low','relative']
    product = itertools.product(band_names, ch_names)
    features = [ feature[1] + '_' + feature[0] for feature in product  ]
    features.append('alpha_frontal')
    features.append('alpha_temporal')
    features.append('alpha_asymetries')
    features.append('beta_frontal')
    features.append('beta_temporal')
    return features

#### 0. Running Parameter ####
time_start = time.time()
dataset_path = "3-DoubleCrossValidation/data"
sampling_rate=125#Hz
start_minute=0
stop_minute=5
segment_second=30
bands = {'delta': [1,3],
    'theta': [4,7],
    'alpha': [8,12],
    'beta': [13,30],
    'gamma': [25,43],
    'slow': [4,13],
    'beta_low': [13,17]
    }
log10 = True
config_string = f"{sampling_rate}-{start_minute}-{stop_minute}-{segment_second}-{log10}"
reset_cache = True

if(os.path.exists('cache') == False):
    # If the folder cache does not exists
    from pathlib import Path
    os.mkdir('cache')
    Path('cache/.keep').touch()

if(os.path.exists(f"cv_results/{config_string}") == False):
    os.mkdir(f"cv_results/{config_string}")

if(reset_cache == True):
    clear_cache(path=f"cache/{config_string}")
    clear_cache(path=f"cv_results/{config_string}")

init_logger(name='main', filename=f'cv_results/{config_string}/main.log')
init_logger(name='ml', filename=f'cv_results/{config_string}/ml.log')
init_logger(name='stat', filename=f'cv_results/{config_string}/stat.log')

# Start
logger = logging.getLogger('main')
logger.info("====== START ======")
logger.info(f"""PARAMETERS:
        dataset_path:{dataset_path}
        sampleing_date:{sampling_rate}
        start_minute:{start_minute}
        stop_minute:{stop_minute}
        segment_second:{segment_second}
        bands:{bands}
        log10:{log10}
        config_string:{config_string}
        reset_cache:{reset_cache}
""")
data, labels, groups = np.array([]), np.array([]), np.array([])
try:
    data = load_cache(path=f"cache/{config_string}", filename='data.pickle')
    labels = load_cache(path=f"cache/{config_string}", filename='labels.pickle')
    groups = load_cache(path=f"cache/{config_string}", filename='gourps.pickle')
    logger.info(f"data,labels,groups is loaded from { f'cache/{config_string}/' }")
except FileNotFoundError as e:
    logger.info(f"cache/{config_string}/ not found. Recalculate preprocessing.")
    #### 1. LOAD DATA ####
    dataset = Dataset_Builder(dataset_path=dataset_path)\
                .with_sampling_rate(sampling_rate=sampling_rate)\
                .with_marker(start_minute=start_minute, 
                        stop_minute=stop_minute, 
                        segment_second=segment_second)\
                .build()
    # data: is in the form of (n_samples, n_channels, n_times) and will need to be processed futhur
    # labels: is the answer. Whether the sample belong to `stressed group` or `non-stressed` group. 
    # groups: is the participant id. this will ensure in the training process that we do not contaminate data of training and validating
    data, labels, groups = dataset.load_data_all()
    
    #### 2. PREPROCESS DATA ####
    data = preprocessing(data=data,sampling_rate=sampling_rate,bands=bands,log10=log10)
    
    logger.info(f"Preprocess result: {data.shape=}")

    save_cache(data=data, path=f"cache/{config_string}", filename='data.pickle')
    save_cache(data=labels, path=f"cache/{config_string}", filename='labels.pickle')
    save_cache(data=groups, path=f"cache/{config_string}", filename='groups.pickle')
except Exception as e:
    logger.error(f"{e}")


logger.info(f"{data.shape=}|{labels.shape=}|{groups.shape=}")

#### 3. STATISTICAL TEST ####
is_normals = check_normality(data)
# if(is_normals.all())
p_values = rank_mannwhitneyu(data, labels)
feature_names = get_feature_name()
#### 4. TRAIN MODEL ####
ranked = p_values.argsort()
names = []
for i in range(15):
    cv_scores = train_model(X=data[:,ranked[:i+1]], y=labels, groups=groups, cv_result_prefix=f"cv_results/{config_string}")
    logger.info(f"{i+1}|10-CV={format(  round(cv_scores.mean(),5), '.5f')}|STD={format(  round(cv_scores.std(),5), '.5f')}")
    names.append(feature_names[ranked[i]])
    logger.info(names)

cv_scores = train_model(X=data[:,], y=labels, groups=groups, cv_result_prefix=f"cv_results/{config_string}")
logger.info(f"all|10-CV={format(  round(cv_scores.mean(),5), '.5f')}|STD={format(  round(cv_scores.std(),5), '.5f')}")
logger.info(f"DONE|Time:{time.time() - time_start}")