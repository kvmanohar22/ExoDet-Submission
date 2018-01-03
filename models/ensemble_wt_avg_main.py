import numpy as np
from hyperopt import hp
from sklearn.base import BaseEstimator


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
        y = np.sum(X * self.weights, axis=1)/self.weights.sum(axis=1)
        return np.vstack([1 - y, y]).T


model = AvgModel(powers=(1.2, 0.63, 0.53, 0.5, 0.64, 0.39, 0.27, 1.51, 1.6400000000000001, 1.78, 0.3, 1.06, 1.61), weights=(0.1, 0.25, 0.0, 1.25, 1.3, 0.25, 0.35000000000000003, 0.25, 0.7000000000000001, 1.4500000000000002, 0.5, 0.35000000000000003, 0.9))

nmodels = 13

params_space = {
    'weights': [hp.quniform('modelwt%d' % i, 0, 2, 0.05) for i in xrange(nmodels)],
    'powers': [hp.qloguniform('modelpow%d' % i, np.log(0.25), np.log(2), 0.01) for i in xrange(nmodels)]
}
