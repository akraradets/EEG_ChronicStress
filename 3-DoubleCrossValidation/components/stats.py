# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411

from scipy.stats import shapiro, normaltest, anderson
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

# def _test_anderson(column:np.ndarray) -> bool:
#     # Anderson-Darling Test is a statistical test that can be used to evaluate whether a 
#     # data sample comes from one of among many known data samples, named for Theodore Anderson and Donald Darling.
#     # 
#     # It can be used to check whether a data sample is normal. The test is a modified version 
#     # of a more sophisticated nonparametric goodness-of-fit statistical test called the `Kolmogorov-Smirnov` test.

#     result = anderson(column)
#     p = 0
#     for i in range(len(result.critical_values)):
#         sl, cv = result.significance_level[i], result.critical_values[i]
#         if result.statistic < result.critical_values[i]:
#             print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
#         else:
# 	    	print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
#     # logger.info(f'{i:3d}:shapiro:Sample looks Gaussian (fail to reject H0)')
#     # # else:
#     # #     print('Sample does not look Gaussian (reject H0)')