import numpy as np
from hyperopt import hp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import random_seed

steps = [
    ('log', LogisticRegression(n_jobs=2, random_state=random_seed))
]

model = Pipeline(steps=steps)

params_space = {
    'log__penalty': hp.choice('penalty', ['l1', 'l2']),
    'log__C': hp.loguniform('C', np.log(0.001), np.log(100)),
    'log__max_iter': 300,
    'log__class_weight': 'balanced',
    'log__random_state': random_seed
}