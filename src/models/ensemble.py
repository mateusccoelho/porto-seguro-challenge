# Libs to deal with tabular data
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, f1_score
from lightgbm import LGBMClassifier

# Misc 
import sys
from tqdm import tqdm
from utils import score_test

if(__name__ == '__main__'):
    simple_lgbm = pd.read_csv('../../data/processed/submission_simple_lgbm.csv')
    rus = pd.read_csv('../../data/processed/submission_rus.csv')
    nearmiss = pd.read_csv('../../data/processed/submission_nearmiss.csv')
    tomek = pd.read_csv('../../data/processed/submission_tomek.csv')

    all_preds = pd.concat([simple_lgbm, rus, nearmiss, tomek], axis=0)
    if(sys.argv[1] == 'mean'):
        mean_preds = all_preds.groupby('id')['predicted'].mean().reset_index()
        mean_preds['predicted'] = np.round(mean_preds['predicted']).astype(int)
        final_preds = mean_preds
    elif(sys.argv[1] == 'max'):
        final_preds = all_preds.groupby('id')['predicted'].max().reset_index().astype(int)
    final_preds.to_csv('../../data/processed/submission_ensemble.csv', index=False)