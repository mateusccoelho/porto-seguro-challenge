# Libs to deal with tabular data
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier

# Misc 
from tqdm import tqdm
from utils import score_test

if(__name__ == '__main__'):
    # combining datasets
    train = pd.read_csv('../../data/raw/train.csv')
    X_train = train.drop(columns=['id', 'y'])
    X_train['target'] = 0
    test = pd.read_csv('../../data/raw/test.csv')
    X_test = test.drop(columns=['id'])
    X_test['target'] = 1
    dataset = pd.concat([X_train, X_test], axis=0)
    X_dataset = dataset.drop(columns=['target'])
    y_dataset = dataset['target']

    # splitting dataset, fitting models
    X_train, X_val, y_train, y_val = \
        train_test_split(X_dataset, y_dataset, test_size = 0.33333, stratify = y_dataset)

    lgbm = LGBMClassifier(n_jobs=-1, random_state=42)
    lgbm.fit(X_train, y_train)
    score = roc_auc_score(y_val, lgbm.predict_proba(X_val)[:,1])
    print('AUC score')
    print(score)