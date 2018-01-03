import numpy as np
from sklearn.base import BaseEstimator


class AvgModel(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        y = np.mean(X, axis=1)
        return np.vstack([1 - y, y]).T

model = AvgModel()
