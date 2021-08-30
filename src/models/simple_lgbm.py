# Libs to deal with tabular data
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, f1_score
from lightgbm import LGBMClassifier

# Misc 
from tqdm import tqdm
from utils import score_test

if(__name__ == '__main__'):
    train = pd.read_csv('../../data/raw/train.csv')
    X_train = train.drop(columns=['id', 'y'])
    y_train = train['y']

    # Cross validation, fitting models
    models_list = []
    cv_scores = []
    cv = StratifiedKFold(5, random_state = 42, shuffle=True)
    for array_idxs in tqdm(cv.split(X_train, y_train)):
        train_index, val_index = array_idxs[0], array_idxs[1]
        X_train_kf, X_val = X_train.loc[train_index], X_train.loc[val_index]
        y_train_kf, y_val = y_train.loc[train_index], y_train.loc[val_index]
        clf = LGBMClassifier(random_state=42, n_jobs=-1, importance_type='gain', is_unbalance=True).fit(X_train_kf, y_train_kf)
        cv_scores.append(f1_score(y_val, clf.predict(X_val)))
        models_list.append(clf)
    print('CV scores')
    print(cv_scores)
    print('Mean CV score')
    print(pd.Series(cv_scores).mean())

    test = pd.read_csv('../../data/raw/test.csv')
    score_test(models_list, test, '../../data/processed/submission_simple_lgbm.csv')