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
        y = np.mean(X * self.weights, axis=1)
        y = np.minimum(y, 1)
        y = np.maximum(y, 0)
        y[y>0.5] = 1
	y[y<0.5] = 0
        return np.vstack([1 - y, y]).T


model = AvgModel(powers=(0.31, 0.56, 0.28500000000000003, 0.295, 0.775, 0.275, 1.95, 1.565, 1.84, 1.8, 0.8300000000000001, 0.42, 0.28), weights=(0.13223893821479152, 0.05406937580988559, 0.339237088343761, 1.9249784460590407, 0.44528060388564394, 0.3615346297000538, 0.6160928261262246, 1.9102800606977988, 0.865224166934328, 1.3093135863057994, 1.2435877533293775, 1.750548900807819, 1.871161399015405))

nmodels = 13

params_space = {
    'weights': [hp.uniform('modelwt%d' % i, 0, 2) for i in xrange(nmodels)],
    'powers': [hp.qloguniform('modelpow%d' % i, np.log(0.25), np.log(2), 0.005) for i in xrange(nmodels)]
}
