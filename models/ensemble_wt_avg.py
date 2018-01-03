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
        print len(self.weights)
        # assert nmodels == len(self.weights) == len(self.powers)
        self.weights = np.array(self.weights).reshape(1, nmodels)
        self.powers = np.array(self.weights).reshape(1, nmodels)

    def predict_proba(self, X):
        X = np.power(X, self.powers)
        y = np.mean(X * self.weights, axis=1)
        y = np.minimum(y, 1)
        y = np.maximum(y, 0)
        return np.vstack([1 - y, y]).T


model = AvgModel(powers=(0.79, 1.05, 1.08, 1.37, 0.35000000000000003, 0.45, 0.58, 0.25, 0.72, 0.4, 1.24, 0.53), weights=(0.7000000000000001, 0.2, 0.45, 2.0, 1.5, 0.05, 1.4500000000000002, 0.65, 1.75, 1.6500000000000001, 1.85, 0.35000000000000003))

nmodels = 12

params_space = {
    'weights': [hp.quniform('modelwt%d' % i, 0, 2, 0.05) for i in xrange(nmodels)],
    'powers': [hp.qloguniform('modelpow%d' % i, np.log(0.25), np.log(2), 0.01) for i in xrange(nmodels)]
}
