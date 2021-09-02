# Libs to deal with tabular data
import numpy as np
import pandas as pd

# Plotting packages
import seaborn as sns
sns.axes_style("darkgrid")
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, f1_score
from imblearn.under_sampling import NearMiss
from lightgbm import LGBMClassifier

# Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_contour, plot_optimization_history
from optuna.visualization import plot_param_importances, plot_slice

# Misc 
import sys
from tqdm import tqdm
from utils import score_test

def lgbm_objective(trial):
    cv_scores = []
    cv = StratifiedKFold(5, random_state = 42, shuffle=True)
    for X_train, y_train, X_val, y_val in fold_sets:
        clf = LGBMClassifier(
            random_state=42, 
            n_jobs=-1,
            is_unbalance = trial.suggest_categorical('is_unbalance', [False, True]),
            num_leaves = trial.suggest_int('num_leaves', 32, 32),
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1.0),
            n_estimators = trial.suggest_int('n_estimators', 100, 100)
        ).fit(X_train_res, y_train_res)
        cv_scores.append(f1_score(y_val, clf.predict(X_val)))
    print(cv_scores)
    return pd.Series(cv_scores).mean()

if(__name__ == '__main__'):
    train = pd.read_csv('../../data/raw/train.csv')
    X_train = train.drop(columns=['id', 'y'])
    y_train = train['y']
    
    fold_sets = []
    cv = StratifiedKFold(5, random_state = 42, shuffle=True)
    for array_idxs in cv.split(X_train, y_train):
        train_index, val_index = array_idxs[0], array_idxs[1]
        X_train_kf, X_val = X_train.loc[train_index], X_train.loc[val_index]
        y_train_kf, y_val = y_train.loc[train_index], y_train.loc[val_index]
        if(sys.argv[1] == '1'):
            sampler = NearMiss(sampling_strategy = 0.5, n_neighbors = 3)
        elif(sys.argv[1] == '2'):
            sampler = NearMiss(sampling_strategy = 0.25, n_neighbors = 3)
        else:
            sampler = NearMiss(
                n_neighbors_ver3 = 9, 
                sampling_strategy = 0.5,
                n_neighbors = 9
            )
        X_train_res, y_train_res = sampler.fit_resample(X_train_kf, y_train_kf)
        fold_sets.append((X_train_res, y_train_res, X_val, y_val))
    
    study = optuna.create_study(sampler=TPESampler(seed = 42), direction='maximize')
    study.optimize(lgbm_objective, n_trials=5)

    print('Best model')
    print('Mean validation F1: ', study.best_value, '\n')
    print(study.best_params)

    models_list = []
    cv_scores = []
    cv = StratifiedKFold(5, random_state = 42, shuffle=True)
    for X_train, y_train, X_val, y_val in fold_sets:
        clf = LGBMClassifier(
            random_state=42, 
            n_jobs=-1,
            is_unbalance = study.best_params['is_unbalance'],
            num_leaves = study.best_params['num_leaves'],
            learning_rate = study.best_params['learning_rate'],
            n_estimators = study.best_params['n_estimators']
        ).fit(X_train_res, y_train_res)
        cv_scores.append(f1_score(y_val, clf.predict(X_val)))
        models_list.append(clf)

    print('Cross validation after tuning')    
    print(pd.Series(cv_scores).mean())

    test = pd.read_csv('../../data/raw/test.csv')
    score_test(models_list, test, f'../../data/processed/submission_nearmiss_v{sys.argv[1]}.csv')