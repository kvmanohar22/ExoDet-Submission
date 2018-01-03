from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import random_seed
from utils.python_utils import quniform_int

steps = [
    ('rf', RandomForestClassifier(n_estimators=1000, n_jobs=2, random_state=random_seed))
]

model = Pipeline(steps=steps)

params_space = {
    'rf__max_depth': quniform_int('max_depth', 2, 30, 1),
    'rf__criterion': hp.choice('criterion', ["gini", "entropy"]),
    'rf__n_estimators': quniform_int('n_estimators', 1000, 10000, 50),
    'rf__min_samples_split': quniform_int('min_samples_split', 2, 100, 1),
}
