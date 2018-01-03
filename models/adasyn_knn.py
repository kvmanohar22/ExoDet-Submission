import numpy as np
from hyperopt import hp
from sklearn.neighbors import KNeighborsClassifier
from config import random_seed
from utils.python_utils import quniform_int
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
steps = [
    ('oversampler', ADASYN(random_state = random_seed)),
    ('knn' ,KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    
]

model = Pipeline(steps=steps)
params_space = {
    'knn__n_neighbors': quniform_int('n_neighbors', 1, 50, 2),
    'knn__weights': hp.choice('weights', ['uniform', 'distance']) ,
    'knn__p': hp.quniform('p', 2.5, 5.5, 1),
}

