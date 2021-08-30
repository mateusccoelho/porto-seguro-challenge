# Libs to deal with tabular data
import numpy as np
import pandas as pd

# Machine Learning
from lightgbm import LGBMClassifier

# Misc 
from tqdm import tqdm

def score_test(models, test, sub_path):
    X_test = test.drop(columns=['id'])
    submission = test[['id']]

    if(not isinstance(models, list)):
        target = model.predict(X_test, num_iteration=model.best_iteration_)

    # Scoring ensemble
    target = np.zeros(len(X_test))
    for model in models:
        pred = model.predict(X_test, num_iteration=model.best_iteration_)
        target += pred / len(models)

    submission = submission.assign(predicted = np.round(target)).astype('int')
    submission.to_csv(sub_path, index=False)