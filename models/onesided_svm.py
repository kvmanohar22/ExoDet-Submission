import numpy as np
from sklearn.svm import SVC
from hyperopt import hp
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import OneSidedSelection

from config import random_seed
from utils.python_utils import quniform_int

steps = [
    ('undersampler', OneSidedSelection(random_state = random_seed)),
    ('SVC', SVC(C = 1, kernel = 'linear', random_state = random_seed, probability =True) )
]
model = Pipeline(steps=steps)

params_space = {
	'svm__C' : hp.quniform('C', 1, 100, 5)
}
