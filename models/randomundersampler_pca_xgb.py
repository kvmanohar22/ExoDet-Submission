import numpy as np
import xgboost as xgb
from hyperopt import hp
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from config import random_seed
from utils.python_utils import quniform_int

steps = [
    ('undersampler', RandomUnderSampler(random_state = random_seed)),
    ('pca', PCA(n_components=50,  random_state=random_seed)),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, silent=True, nthread=3, seed=random_seed))
]

model = Pipeline(steps=steps)

params_space = {
    'pca__n_components': quniform_int('n_components', 20, 200, 10),
    'xgb__max_depth': quniform_int('max_depth', 10, 30, 1),
    'xgb__min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'xgb__subsample': hp.uniform('subsample', 0.8, 1),
    'xgb__n_estimators': quniform_int('n_estimators', 1000, 10000, 50),
    'xgb__learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001,
    'xgb__gamma': hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'xgb__colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
}
