import unittest

import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import tree

import starboost as sb


class TestClassificationExamples(unittest.TestCase):

    def test_sklearn_log_loss(self):
        """Tests against the output of scikit-learn's GradientBoostingClassifier using logloss."""
        X, y = datasets.load_breast_cancer(return_X_y=True)

        star = sb.BoostingClassifier(
            loss=sb.loss.LogLoss(),
            base_estimator=tree.DecisionTreeRegressor(max_depth=3),
            n_estimators=30,
            learning_rate=0.1,
        )
        star = star.fit(X, y)

        scikit = ensemble.GradientBoostingClassifier(
            loss='deviance',
            max_depth=3,
            n_estimators=30,
            learning_rate=0.1,
        )
        scikit = scikit.fit(X, y)

        for y1, y2 in zip(star.predict_proba_iter(X), scikit.staged_predict_proba(X)):
            self.assertTrue(np.mean(np.abs(y1 - y2)) < 1e-5)
