import numpy as np
import logging
import os

from components.ml import train_model
from components.utils import save_cache, load_cache

def forward_selection(data:np.ndarray, labels:np.ndarray, groups:np.ndarray, cv_result_prefix:str="selection"):
    logger = logging.getLogger('selection')
    
    path_cache = f'{cv_result_prefix}/forward'
    if(os.path.exists(path_cache) == False):
        os.mkdir(path_cache)

    try:
        rank = load_cache(path=path_cache, filename='rank.pickle')
    except FileNotFoundError as e:
        rank = []
    except Exception as e:
        logger.error(e)
        raise e

    # selecting 25 features
    count = len(rank)
    while count < 25:
        new_feature = _forward_selection_next_feature(rank, data, labels, groups, cv_result_prefix)
        rank.append(new_feature)
        save_cache(data=rank, path=path_cache, filename=f"{count}.pickle")
        save_cache(data=rank, path=path_cache, filename=f"rank.pickle")
    return rank


def _forward_selection_next_feature(base_feature:list, data:np.ndarray, labels:np.ndarray, groups:np.ndarray, cv_result_prefix:str="selection") -> int:
    logger = logging.getLogger('selection')

    features_to_search = set(range(data.shape[1])).difference(base_feature)
    best_score = 0
    best_std = 0
    best_set = set({})

    for index in features_to_search:
        temp_set = set(base_feature).union([index])
        cv_scores = train_model(X=data[:,np.array(list(temp_set))], y=labels, groups=groups, cv_result_prefix=f"{cv_result_prefix}")
        logger.info(f"{temp_set}|10-CV={format(  round(cv_scores.mean(),5), '.5f')}|STD={format(  round(cv_scores.std(),5), '.5f')}")
        score, std = cv_scores.mean(), cv_scores.std()

        if(score > best_score or (score == best_score and std < best_std)):
            best_score = score
            best_std = std
            best_set = temp_set

    new_feature = best_set.difference(base_feature).pop()
    
    return new_feature # type: ignore