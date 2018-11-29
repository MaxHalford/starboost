import abc

import numpy as np
from sklearn import base


class SimpleEstimator(abc.ABC, base.BaseEstimator, base.RegressorMixin):

    @abc.abstractmethod
    def make_estimate(self, y):
        pass

    def fit(self, X, y):
        self.estimate_ = self.make_estimate(y)
        return self

    def predict(self, X):
        n, k = len(X), len(self.estimate_)
        return np.full(fill_value=self.estimate_, shape=(n, k), dtype=np.float32)


class MeanEstimator(SimpleEstimator):

    def make_estimate(self, y):
        return y.mean(axis=0)


class QuantileEstimator(SimpleEstimator):

    def __init__(self, alpha):
        self.alpha = alpha

    def make_estimate(self, y):
        return np.quantile(y, q=self.alpha, axis=0)


class PriorProbabilityEstimator(SimpleEstimator):

    def make_estimate(self, y):
        return np.sum(y, axis=0) / len(y)
