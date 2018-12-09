import unittest

import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import tree
from sklearn import utils

import starboost as sb


class ScikitLearnL1Loss(sb.losses.L1Loss):
    """scikit-learn does two things different than StarBoost for L1 regression:

    - the zeros from the gradient are replaced with -1s
    - their weighted percentile isn't exactly equal to the median when there are no sample weights
    """

    def gradient(self, y_true, y_pred):
        sign = np.sign(y_pred - y_true)
        np.place(sign, sign==0, -1)
        return sign

    @property
    def tree_line_searcher(self):

        def update_leaf(y_true, y_pred, gradient):
            residual = y_true - y_pred
            return utils.stats._weighted_percentile(residual, np.ones_like(residual), percentile=50)

        return sb.line_searchers.LeafLineSearcher(update_leaf=update_leaf)


class TestBoostingRegressor(unittest.TestCase):

    def test_explained_ai_l2_loss_section_2_4(self):
        """Reproduces the results from https://explained.ai/gradient-boosting/L2-loss.html#sec:2.4"""
        X = [[750], [800], [850], [900], [950]]
        y = [1160, 1200, 1280, 1450, 2000]
        y_preds = [
            [1418, 1418, 1418, 1418, 1418],
            [1272.5, 1272.5, 1272.5, 1272.5, 2000],
            [1180, 1180, 1334.2, 1334.2, 2061.7],
            [1195.4, 1195.4, 1349.6, 1349.6, 2000]
        ]

        model = sb.BoostingRegressor(
            loss=sb.losses.L2Loss(),
            base_estimator=tree.DecisionTreeRegressor(max_depth=1),
            tree_flavor=True,
            n_estimators=3,
            learning_rate=1.0,
        )
        model.fit(X, y)

        for i, y_pred in enumerate(model.iter_predict(X, include_init=True)):
            np.testing.assert_allclose(y_preds[i], y_pred, atol=0.1)

    def test_explained_ai_l1_loss_section_1_3(self):
        """Reproduces the results from https://explained.ai/gradient-boosting/L1-loss.html#sec:1.3"""
        X = [[750], [800], [850], [900], [950]]
        y = [1160, 1200, 1280, 1450, 2000]
        y_preds = [
            [1280, 1280, 1280, 1280, 1280],
            [1180, 1180, 1450, 1450, 1450],
            [1160, 1190, 1460, 1460, 1460],
            [1155, 1185, 1455, 1455, 2000]
        ]

        model = sb.BoostingRegressor(
            loss=sb.losses.L1Loss(),
            base_estimator=tree.DecisionTreeRegressor(max_depth=1),
            tree_flavor=True,
            n_estimators=3,
            learning_rate=1.0,
        )
        model.fit(X, y)

        for i, y_pred in enumerate(model.iter_predict(X, include_init=True)):
            np.testing.assert_allclose(y_preds[i], y_pred, atol=0.1)

    def test_sklearn_l2(self):
        """Tests against the output of scikit-learn's GradientBoostingRegressor using MSE."""
        X, y = datasets.load_boston(return_X_y=True)

        star = sb.BoostingRegressor(
            loss=sb.losses.L2Loss(),
            base_estimator=tree.DecisionTreeRegressor(max_depth=3, random_state=42),
            tree_flavor=True,
            n_estimators=30,
            learning_rate=0.1,
        )
        star = star.fit(X, y)

        scikit = ensemble.GradientBoostingRegressor(
            loss='ls',
            max_depth=3,
            n_estimators=30,
            learning_rate=0.1,
            random_state=42
        )
        scikit = scikit.fit(X, y)

        for y1, y2 in zip(star.iter_predict(X), scikit.staged_predict(X)):
            self.assertTrue(np.mean(np.abs(y1 - y2)) < 1e-5)

    def test_sklearn_l1(self):
        """Tests against the output of scikit-learn's GradientBoostingRegressor using MAE."""
        X, y = datasets.load_boston(return_X_y=True)

        star = sb.BoostingRegressor(
            loss=ScikitLearnL1Loss(),
            base_estimator=tree.DecisionTreeRegressor(max_depth=2, random_state=42),
            tree_flavor=True,
            n_estimators=5,
            learning_rate=0.1,
        )
        star = star.fit(X, y)

        scikit = ensemble.GradientBoostingRegressor(
            loss='lad',
            max_depth=2,
            n_estimators=5,
            learning_rate=0.1,
            random_state=42
        )
        scikit = scikit.fit(X, y)

        for y1, y2 in zip(star.iter_predict(X), scikit.staged_predict(X)):
            self.assertTrue(np.mean(np.abs(y1 - y2)) < 1e-5)
