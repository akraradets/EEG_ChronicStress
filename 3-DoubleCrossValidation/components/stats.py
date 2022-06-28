# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411

from scipy.stats import shapiro, normaltest, anderson
from scipy.stats import mannwhitneyu
import numpy as np
import logging

def check_normality(data:np.ndarray) -> np.ndarray:
    logger = logging.getLogger('stat')
    is_normals = []
    for i in range(data.shape[1]):
        column = data[:, i]
        is_normal_shapiro = _test_shapiro(column)
        is_normal_normal = _test_normaltest(column)
        if(is_normal_shapiro or is_normal_normal):
            logger.info(f"{i:3d}|shapiro={is_normal_shapiro}|normal={is_normal_normal}")
        is_normals.append(is_normal_shapiro and is_normal_normal)
    is_normals = np.array(is_normals)
    logger.info(f"normal_column:{np.where(is_normals)}")
    return is_normals

def _test_shapiro(column:np.ndarray) -> bool:
    alpha = 0.05
    stat, p = shapiro(column)
    if(p > alpha):
        return True
    return False

def _test_normaltest(column:np.ndarray) -> bool:
    alpha = 0.05
    stat, p = normaltest(column)
    if(p > alpha):
        return True
    return False

def rank_mannwhitneyu(data:np.ndarray, labels:np.ndarray) -> np.ndarray:
    p_values = []
    for i in range(133):
        result = mannwhitneyu(data[labels == 0, i], data[labels == 1, i])
        p_values.append(result.pvalue)
    return np.array(p_values)