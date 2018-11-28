import numpy as np
from sklearn import base
from sklearn import utils


class MeanEstimator(base.BaseEstimator, base.RegressorMixin):

    def fit(self, X, y):
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return np.full(fill_value=self.mean_, shape=len(X))


class QuantileEstimator(base.BaseEstimator, base.RegressorMixin):

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        self.quantile_ = utils.stats._weighted_percentile(y, sample_weight, self.alpha * 100.0)
        return self

    def predict(self, X):
        return np.full(fill_value=self.quantile_, shape=len(X), dtype=np.float32)


class LogOdds(base.BaseEstimator, base.RegressorMixin):

    def __init__(self, scale=1.):
        self.scale = scale

    def fit(self, X, y):
        pos = np.sum(y)
        neg = np.sum(1 - y)
        self.prior_ = self.scale * np.log(pos / neg)
        return self

    def predict(self, X):
        return np.full(fill_value=self.prior_, shape=len(X), dtype=np.float32)
