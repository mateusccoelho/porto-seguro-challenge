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

def nearmiss_objective(trial):
    cv_scores = []
    cv = StratifiedKFold(5, random_state = 42, shuffle=True)
    for array_idxs in cv.split(X_train, y_train):
        train_index, val_index = array_idxs[0], array_idxs[1]
        X_train_kf, X_val = X_train.loc[train_index], X_train.loc[val_index]
        y_train_kf, y_val = y_train.loc[train_index], y_train.loc[val_index]
        
        version = int(sys.argv[1])
        if(version == 3):
            n_neighbors_ver3 = trial.suggest_categorical('n_neighbors_ver3', [3,5,7,9])
        else:
            n_neighbors_ver3 = 3 # default
        rus = NearMiss(
            sampling_strategy = trial.suggest_float('sampling_strategy', 0.253, 1),
            version = version,
            n_neighbors = trial.suggest_categorical('n_neighbors', [3,5,7,9]),
            n_neighbors_ver3 = n_neighbors_ver3 
        )
        X_train_res, y_train_res = rus.fit_resample(X_train_kf, y_train_kf)
        clf = LGBMClassifier(random_state=42, n_jobs=-1).fit(X_train_res, y_train_res)
        cv_scores.append(f1_score(y_val, clf.predict(X_val)))
    return pd.Series(cv_scores).mean()

if(__name__ == '__main__'):
    train = pd.read_csv('../../data/raw/train.csv')
    X_train = train.drop(columns=['id', 'y'])
    y_train = train['y']
    
    study = optuna.create_study(sampler=TPESampler(seed = 42), direction='maximize')
    study.optimize(nearmiss_objective, n_trials=100)

    print('Best model')
    print('Mean validation F1: ', study.best_value, '\n')
    print(study.best_params)

    models_list = []
    cv_scores = []
    cv = StratifiedKFold(5, random_state = 42, shuffle=True)
    for array_idxs in tqdm(cv.split(X_train, y_train)):
        train_index, val_index = array_idxs[0], array_idxs[1]
        X_train_kf, X_val = X_train.loc[train_index], X_train.loc[val_index]
        y_train_kf, y_val = y_train.loc[train_index], y_train.loc[val_index]
        
        if('n_neighbors_ver3' in study.best_params):
            n_neighbors_ver3 = study.best_params['n_neighbors_ver3']
        else:
            n_neighbors_ver3 = 3 # default 
        rus = NearMiss(
            sampling_strategy = study.best_params['sampling_strategy'],
            version = int(sys.argv[1]),
            n_neighbors = study.best_params['n_neighbors'],
            n_neighbors_ver3 = n_neighbors_ver3
        )
        X_train_res, y_train_res = rus.fit_resample(X_train_kf, y_train_kf)
        clf = LGBMClassifier(
            random_state=42, 
            n_jobs=-1, 
            importance_type='gain'
        ).fit(X_train_res, y_train_res)
        cv_scores.append(f1_score(y_val, clf.predict(X_val)))
        models_list.append(clf)

    print('Cross validation after tuning')    
    print(pd.Series(cv_scores).mean())

    test = pd.read_csv('../../data/raw/test.csv')
    score_test(models_list, test, f'../../data/processed/submission_nearmiss_v{sys.argv[1]}.csv')