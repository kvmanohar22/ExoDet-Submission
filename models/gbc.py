from hyperopt import hp
from sklearn.ensemble import GradientBoostingClassifier
from config import random_seed
import numpy as np

from utils.python_utils import quniform_int

model = GradientBoostingClassifier(learning_rate=0.03, max_depth=7, max_features=0.45, n_estimators=690,
                                   min_samples_leaf=14, verbose=1, random_state=random_seed)

params_space = {
    "n_estimators": quniform_int("n_estimators", 100, 5000, 10),
    "learning_rate": hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "max_depth": quniform_int("max_depth", 1, 20, 1),
    "min_samples_leaf": quniform_int("min_samples_leaf", 1, 15, 1),
    "random_state": random_seed,
    "verbose": 1,
}
