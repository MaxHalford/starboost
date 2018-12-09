import unittest

import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import tree

import starboost as sb


class ScikitLearnLogOdds():

    def fit(self, X, y):
        pos = np.sum(y, axis=0)
        neg = np.sum(1 - y, axis=0)
        self.prior_ = np.log(pos / neg)
        return self

    def predict(self, X):
        return np.full(fill_value=self.prior_, shape=(len(X), len(self.prior_)))


class TestBoostingClassifier(unittest.TestCase):

    def test_sklearn_log_loss(self):
        """Tests against the output of scikit-learn's GradientBoostingClassifier using logloss."""
        X, y = datasets.load_breast_cancer(return_X_y=True)

        star = sb.BoostingClassifier(
            loss=sb.losses.LogLoss(),
            init_estimator=ScikitLearnLogOdds(),
            base_estimator=tree.DecisionTreeRegressor(max_depth=3, random_state=42),
            tree_flavor=True,
            n_estimators=30,
            learning_rate=0.1,
        )
        star = star.fit(X, y)

        scikit = ensemble.GradientBoostingClassifier(
            loss='deviance',
            max_depth=3,
            n_estimators=30,
            learning_rate=0.1,
            random_state=42
        )
        scikit = scikit.fit(X, y)

        for y1, y2 in zip(star.iter_predict_proba(X), scikit.staged_predict_proba(X)):
            self.assertTrue(np.mean(np.abs(y1 - y2)) < 1e-4)
