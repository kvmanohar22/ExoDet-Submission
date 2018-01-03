import numpy as np
import xgboost as xgb
from hyperopt import hp
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from config import random_seed
from utils.python_utils import quniform_int

steps = [
    ('tsne', TSNE(n_components=4, learning_rate=200, random_state=random_seed)),
    ('xgb', xgb.XGBClassifier(n_estimators=5900, max_depth=11, min_child_weight=4, subsample=0.932626370862, gamma=0.7, colsample_bytree=0.85, learning_rate=0.125, silent=True, nthread=1, seed=random_seed))
]

model = Pipeline(steps=steps)

params_space = {
    'tsne__n_components': quniform_int('n_components', 2, 100, 2),
    'tsne__learning_rate': quniform_int('learning_rate', 10, 1000, 10),
    'xgb__max_depth': quniform_int('max_depth', 10, 30, 1),
    'xgb__min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'xgb__subsample': hp.uniform('subsample', 0.8, 1),
    'xgb__n_estimators': quniform_int('n_estimators', 1000, 10000, 50),
    'xgb__learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001,
    'xgb__gamma': hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'xgb__colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
}

