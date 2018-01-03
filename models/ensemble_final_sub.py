import numpy as np
from hyperopt import hp
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class AvgModel(BaseEstimator):
    def __init__(self, weights=None, powers=None):
        self.weights = weights
        self.powers = powers

    def fit(self, X, y):
        self.classes_ = set(y)
        _, nmodels = X.shape
        # assert nmodels == len(self.weights) == len(self.powers)
        self.weights = np.array(self.weights).reshape(1, nmodels)
        self.powers = np.array(self.weights).reshape(1, nmodels)

    def predict_proba(self, X):
        X = np.power(X, self.powers)
        y = np.mean(X * self.weights, axis=1)
        y = np.minimum(y, 1)
        y = np.maximum(y, 0)
        return np.vstack([1 - y, y]).T


# model = AvgModel(powers=(1.2, 0.63, 0.53, 0.5, 0.64, 0.39, 0.27, 1.51, 1.6400000000000001, 1.78, 0.3, 1.06, 1.61), weights=(0.1, 0.25, 0.0, 1.25, 1.3, 0.25, 0.35000000000000003, 0.25, 0.7000000000000001, 1.4500000000000002, 0.5, 0.35000000000000003, 0.9))

steps = [
#    ('scaler', MinMaxScaler()),
    ('ensemble', AvgModel(powers=(0.55, 0.44, 0.68, 1.21, 0.36, 1.3800000000000001, 0.74, 0.66, 1.21, 0.71, 1.03), weights=(0.4, 0.15000000000000002, 0.15000000000000002, 1.8, 0.1, 1.35, 0.30000000000000004, 0.8500000000000001, 1.25, 0.75, 0.0)))
]

model = Pipeline(steps=steps)

nmodels = 11

params_space = {
    'ensemble__weights': [hp.quniform('modelwt%d' % i, 0, 2, 0.05) for i in xrange(nmodels)],
    'ensemble__powers': [hp.qloguniform('modelpow%d' % i, np.log(0.1), np.log(2), 0.01) for i in xrange(nmodels)]
}
