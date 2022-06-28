from ast import ExceptHandler
from signal import raise_signal
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
import pandas as pd
import logging
from time import time
import os
import re
from typing import List

def get_10_cv(file_path:str) -> List[str]:
    lines = []
    with open(file_path, 'r') as f:
        # last_line = f.readlines()[-1]
        for line in f:
            if(re.search('grid.best_params_',line)):
                lines.append(line)
    return lines

def get_accs(file_path) -> List[float]:
    best_rows = get_10_cv(file_path=file_path)
    # 26-06-2022 16:09:10|ml.py:44|INFO|3/10|grid.best_params_={'C': 1000000.0, 'gamma': 1e-09, 'kernel': 'rbf', 'max_iter': 1000}, grid.best_score_=0.6678240740740741, grid.best_index_=28, acc=0.4375, time=2.2673180103302
    accs = []
    for line in best_rows:
        acc = float(line[line.index('acc=') + len('acc='): line.index('time=')-2])
        accs.append(acc)
    return accs

def get_logger_filename(logger_name:str) -> str:
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers:
        if(type(handler) == logging.FileHandler):
            return handler.baseFilename # type: ignore
    else:
        raise Exception(f"logging.FileHandler is not exists in the logger name {logger_name}")

def train_model(X:np.ndarray, y:np.ndarray, groups:np.ndarray, cv_result_prefix:str="") -> np.ndarray:
    logger = logging.getLogger("ml")
    """
        X: shape (n_samples, n_features)
        y: shape (n_samples, )
        groups: shape (n_samples, )
    """

    n_split_outter = 10
    cv_outter = StratifiedGroupKFold(n_splits=n_split_outter, shuffle=False)
    accs = get_accs(get_logger_filename('ml'))
    # if(len(accs) == 10): 
        # return np.array(accs)
    accs=[]

    logger.info(f"X.shape={X.shape}, y.shape={y.shape}, groups.shape={groups.shape}")
    for epoch, (idxs_train, idxs_test) in enumerate(cv_outter.split(X,y,groups)):
        start = time()
        print(f"BEGIN EPOCH: {epoch+1}/{n_split_outter}")
        filename = os.path.join(cv_result_prefix,f"{epoch+1}.csv")
        if(epoch < len(accs)): continue


        X_train, X_test = X[idxs_train], X[idxs_test]
        y_train, y_test = y[idxs_train], y[idxs_test]
        groups_train, groups_test = groups[idxs_train], groups[idxs_test]
        assert set(groups_train).isdisjoint(set(groups_test)),f"Contaminated.\ngroups_train:{groups_train}\ngroups_test:{groups_test}"

        grid = _build_model(X_train,y_train,groups_train)
        # Evaluation
        model = grid.best_estimator_
        predict = model.predict(X_test) # type: ignore
        acc = sum(predict == y_test) / len(y_test)
        accs.append(acc)
        # save csv
        logger.info(f"{epoch+1}/{n_split_outter}|grid.best_params_={grid.best_params_}, grid.best_score_={grid.best_score_}, grid.best_index_={grid.best_index_}, acc={acc}, time={time()-start}" )
        pd.DataFrame(grid.cv_results_).to_csv(filename)
    accs = np.array(accs)
    logger.info(f"final|10-CV={format(  round(accs.mean(),5), '.5f')}|STD={format(  round(accs.std(),5), '.5f')}")
    return accs

def _build_model(X:np.ndarray,y:np.ndarray,groups:np.ndarray) -> GridSearchCV:
    """
        This function will only optimized models
    """
    n_split = 9
    cv = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=42)
    # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    C_range = np.logspace(-2, 10, 7)
    gamma_range = np.logspace(-9, 3, 7)
    tuned_parameters = [
            {"kernel": ["rbf"],    "C": C_range, "max_iter":[1000],  "gamma": gamma_range},
        ]
    grid = GridSearchCV(SVC(), param_grid=tuned_parameters, cv=cv, n_jobs=os.cpu_count(), refit=True, verbose=4, return_train_score=True)
    grid.fit(X=X, y=y, groups=groups)
    return grid