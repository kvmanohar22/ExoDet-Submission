import numpy as np
from hyperopt import hp
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from lpproj import LocalityPreservingProjection 
from sklearn.decomposition import PCA
from config import random_seed
from utils.python_utils import quniform_int
from sklearn.pipeline import Pipeline
steps = [
    ('lpp', LocalityPreservingProjection(n_components=55)),
    ('knn' ,KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
]

model = Pipeline(steps=steps)
params_space = {
    'pca__n_components': quniform_int('n_components', 20, 200, 10),
    'knn__n_neighbors': quniform_int('n_neighbors', 1, 50, 2),
    'knn__weights': hp.choice('weights', ['uniform', 'distance']) ,
    'knn__p': hp.quniform('p', 2.5, 5.5, 1),
}


