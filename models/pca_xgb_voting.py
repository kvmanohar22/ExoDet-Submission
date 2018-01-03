import numpy as np
import xgboost as xgb
from hyperopt import hp
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

from config import random_seed
from utils.python_utils import quniform_int

classifer_1 = Pipeline([
    ('pca', PCA(n_components=51, random_state=random_seed)),
    ('xgb', xgb.XGBClassifier(n_estimators=5900, max_depth=11, min_child_weight=4, subsample=0.932626370862, gamma=0.7,
                              colsample_bytree=0.85, learning_rate=0.125, silent=True, nthread=3, seed=random_seed))
])

classifer_2 = Pipeline([
    ('pca', PCA(n_components=75, random_state=random_seed)),
    ('xgb', xgb.XGBClassifier(n_estimators=5000, max_depth=15, min_child_weight=10, subsample=0.9, gamma=1,
                              colsample_bytree=0.7, learning_rate=0.25, silent=True, nthread=3, seed=random_seed * 2))
])

classifer_3 = xgb.XGBClassifier(n_estimators=5900, max_depth=11, min_child_weight=4, subsample=0.932626370862,
                                gamma=0.7, colsample_bytree=0.85, learning_rate=0.125, silent=True, nthread=1,
                                seed=random_seed)

# classifer_4 = xgb.XGBClassifier(n_estimators=5000, max_depth=15, min_child_weight=10, subsample=0.9, gamma=1, colsample_bytree=0.7, learning_rate=0.25, silent=True, nthread=3, seed=random_seed*2)

steps = [
    ('voting',
     VotingClassifier(estimators=[('pca_xgb_1', classifer_1), ('pca_xgb_2', classifer_2), ('xgb_3', classifer_3)],
                      voting='soft'))
]

model = Pipeline(steps=steps)

params_space = {
    'xgb__max_depth': quniform_int('max_depth', 10, 30, 1),
    'xgb__min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'xgb__subsample': hp.uniform('subsample', 0.8, 1),
    'xgb__n_estimators': quniform_int('n_estimators', 1000, 10000, 50),
    'xgb__learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001,
    'xgb__gamma': hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'xgb__colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
}
