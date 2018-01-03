import numpy as np
import xgboost as xgb
from hyperopt import hp
from sklearn.manifold import LocallyLinearEmbedding
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from config import random_seed
from utils.python_utils import quniform_int

steps = [
    ('lle', LocallyLinearEmbedding(n_components=175, n_jobs=4,  random_state=random_seed)),
    # ('knn', KNeighborsClassifier(n_jobsneighbors=10, n_jobs=4)),
    ('xgb', xgb.XGBClassifier(n_estimators=7250, max_depth=26, min_child_weight=9, subsample=0.921621811501, gamma=0.173667385702, colsample_bytree=0.65, learning_rate=0.0240343154532, silent=True, nthread=3, seed=random_seed))

]
model = Pipeline(steps=steps)

params_space = {
    'lle__n_components': quniform_int('n_components', 10, 250, 5),
    'knn__n_neighbors': quniform_int('n_neighbors', 1, 25, 1),
    'xgb__max_depth': quniform_int('max_depth', 10, 30, 1),
    'xgb__min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'xgb__subsample': hp.uniform('subsample', 0.8, 1),
    'xgb__n_estimators': quniform_int('n_estimators', 1000, 10000, 50),
    'xgb__learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001,
    'xgb__gamma': hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'xgb__colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
}