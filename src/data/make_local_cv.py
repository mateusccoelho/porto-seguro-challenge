# Libs to deal with tabular data
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, f1_score
from imblearn.under_sampling import NearMiss
from lightgbm import LGBMClassifier

# Misc 
import sys
from tqdm import tqdm

train = pd.read_csv('../../data/raw/train.csv')
X_train = train.drop(columns=['id', 'y'])
y_train = train['y']

fold_sets = []
cv = StratifiedKFold(5, random_state = 42, shuffle=True)
for idx, array_idxs in enumerate(cv.split(X_train, y_train)):
    train_index, val_index = array_idxs[0], array_idxs[1]
    train.loc[val_index, 'cv'] = idx

train.to_csv('../../data/interim/local_cv.csv', index=False)
    