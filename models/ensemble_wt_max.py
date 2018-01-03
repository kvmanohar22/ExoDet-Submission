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
        assert nmodels == len(self.weights) == len(self.powers)
        self.weights = np.array(self.weights).reshape(1, nmodels)
        self.powers = np.array(self.weights).reshape(1, nmodels)

    def predict_proba(self, X):
        X = np.power(X, self.powers)
        y = np.max(X * self.weights, axis=1)
        y = np.minimum(y, 1)
        y = np.maximum(y, 0)
        return np.vstack([1 - y, y]).T


model = AvgModel()

nmodels = 12

params_space = {
    'weights': [hp.quniform('modelwt%d' % i, 0, 2, 0.05) for i in xrange(nmodels)],
    'powers': [hp.qloguniform('modelpow%d' % i, np.log(0.25), np.log(2), 0.01) for i in xrange(nmodels)]
}
