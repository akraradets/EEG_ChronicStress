from components.dataset import Dataset_Builder
from components.preprocessing import preprocessing
from components.ml import train_model
from components.logger import init_logger
from components.stats import check_normality
from components.utils import *
from sklearn.preprocessing import StandardScaler
import logging 
import time

#### 0. Running Parameter ####
time_start = time.time()
dataset_path = "3-DoubleCrossValidation/data"
sampling_rate=125#Hz
start_minute=1
stop_minute=3
segment_second=15
config_string = f"{sampling_rate}-{start_minute}-{stop_minute}-{segment_second}"
reset_cache = False

if(os.path.exists('cache') == False):
    # If the folder cache does not exists
    from pathlib import Path
    os.mkdir('cache')
    Path('cache/.keep').touch()

if(os.path.exists(f"cv_results/{config_string}") == False):
    os.mkdir(f"cv_results/{config_string}")

if(reset_cache == True):
    clear_cache(path=f"cache/{config_string}")

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
        config_string:{config_string}
        reset_cache:{reset_cache}
""")

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
logger.info(f"{labels=}")
logger.info(f"{groups=}")
logger.info(f"{data.shape=}|{labels.shape=}|{groups.shape=}")

#### 2. PREPROCESS DATA ####
target_file = 'preprocess.pickle'
try:
    data = load_cache(path=f"cache/{config_string}", filename=target_file)
    logger.info(f"Preprocess is loaded from { f'cache/{config_string}/{target_file}' }")
except FileNotFoundError as e:
    logger.info(f"cache/{config_string}/{target_file} not found. Recalculate preprocessing.")
    data = preprocessing(data=data,sampling_rate=sampling_rate)
    save_cache(data=data, path=f"cache/{config_string}", filename=target_file)
except Exception as e:
    logger.error(f"{e}")

logger.info(f"Preprocess result: {data.shape=}")

# Upon testing, this StandardScaler will apply (X_i - Mu_i) / std_i for each i (column/feature) individually
scaler = StandardScaler()
data = scaler.fit_transform(data)

#### 3. STATISTICAL TEST ####
is_normals = check_normality(data)

#### 4. TRAIN MODEL ####
cv_scores = train_model(X=data, y=labels, groups=groups, cv_result_prefix=f"cv_results/{config_string}")
logger.info(f"final|10-CV={format(  round(cv_scores.mean(),5), '.5f')}|STD={format(  round(cv_scores.std(),5), '.5f')}")
logger.info(f"DONE|Time:{time.time() - time_start}")