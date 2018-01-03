import numpy as np
import xgboost as xgb
from hyperopt import hp
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours

from config import random_seed
from utils.python_utils import quniform_int

steps = [
    ('undersampler', EditedNearestNeighbours(random_state=random_seed, n_neighbors=3)),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, silent=True, nthread=3, seed=random_seed))
]

model = Pipeline(steps=steps)

params_space = {
    'undersampler__n_neighbors': quniform_int('n_neighbors', 2, 10, 1),
    'xgb__max_depth': quniform_int('max_depth', 10, 30, 1),
    'xgb__min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'xgb__subsample': hp.uniform('subsample', 0.8, 1),
    'xgb__n_estimators': quniform_int('n_estimators', 1000, 10000, 50),
    'xgb__learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001,
    'xgb__gamma': hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'xgb__colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
}

model.set_params({
    'xgb__colsample_bytree': 0.55,
    'xgb__n_estimators': 1000,
    'xgb__subsample': 0.81758885827,
    'xgb__min_child_weight': 2.0,
    'xgb__learning_rate': 0.0091861014503,
    'xgb__gamma': 1.19618674618,
    'undersampler__n_neighbors': 7,
    'xgb__max_depth': 21
})
