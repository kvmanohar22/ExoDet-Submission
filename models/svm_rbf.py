from hyperopt import hp
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import random_seed

steps = [
    ('SVC', SVC(C=1, kernel='rbf', random_state=random_seed, probability=True))
]

model = Pipeline(steps=steps)

params_space = {
    'svm__C' : hp.quniform('C', 1, 100, 5)
}
