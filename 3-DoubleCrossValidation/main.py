from components.dataset import Dataset_Builder
from components.preprocessing import preprocessing
from components.utils import *
import logging 
import sys
import time

#### Init logger ####
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s|%(filename)s:%(lineno)d|%(levelname)s|%(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    handlers=[
        logging.FileHandler(filename='running.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("====== START ======")

#### 0. Running Parameter ####
time_start = time.time()
dataset_path = "3-DoubleCrossValidation/data"
sampling_rate=125#Hz
start_minute=1
stop_minute=2
segment_second=5
config_string = f"{sampling_rate}-{start_minute}-{stop_minute}-{segment_second}"
reset_cache = False
logging.info(f"""PARAMETERS:
        dataset_path:{dataset_path}
        sampleing_date:{sampling_rate}
        start_minute:{start_minute}
        stop_minute:{stop_minute}
        segment_second:{segment_second}
        config_string:{config_string}
        reset_cache:{reset_cache}
""")

if(os.path.exists('cache') == False):
    # If the folder cache does not exists
    from pathlib import Path
    os.mkdir('cache')
    Path('cache/.keep').touch()

if(reset_cache == True):
    clear_cache(path=f"cache/{config_string}")

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
logging.info(f"{labels=}")
logging.info(f"{groups=}")
logging.info(f"{data.shape=}|{labels.shape=}|{groups.shape=}")

#### 2. Preprocessing data ####
target_file = 'preprocess.pickle'
try:
    data = load_cache(path=f"cache/{config_string}", filename=target_file)
    logging.info(f"Preprocess is loaded from { f'cache/{config_string}/{target_file}' }")
except FileNotFoundError as e:
    logging.info(f"cache/{config_string}/{target_file} not found. Recalculate preprocessing.")
    data = preprocessing(data=data,sampling_rate=sampling_rate)
    save_cache(data=data, path=f"cache/{config_string}", filename=target_file)
except Exception as e:
    logging.error(f"{e}")

logging.info(f"Preprocess result: {data.shape=}")




logging.info(f"DONE|Time:{time.time() - time_start}")