import abc
import collections
import copy
import itertools
import warnings

import numpy as np
from scipy.special import expit as sigmoid
from sklearn import base
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import utils

from . import losses


warnings.simplefilter('ignore', np.RankWarning)


class BaseBoosting(abc.ABC, ensemble.BaseEnsemble):
    """Implements logic common to all other boosting classes."""
    def __init__(self, loss=None, base_estimator=None, tree_flavor=False, n_estimators=30,
                 init_estimator=None, line_searcher=None, learning_rate=0.1, row_sampling=1.0,
                 col_sampling=1.0, eval_metric=None, early_stopping_rounds=None, random_state=None):
        self.loss = loss or self._default_loss
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.init_estimator = init_estimator or self.loss.default_init_estimator
        self.line_searcher = line_searcher
        if self.line_searcher is None and tree_flavor:
            self.line_searcher = self.loss.tree_line_searcher
        self.learning_rate = learning_rate
        self.row_sampling = row_sampling
        self.col_sampling = col_sampling
        self.eval_metric = eval_metric or loss
        self.early_stopping_rounds = early_stopping_rounds
        self.rng = utils.check_random_state(random_state)

    @abc.abstractmethod
    def _transform_y_pred(self, y_pred):
        pass

    @property
    @abc.abstractmethod
    def _default_loss(self):
        pass

    def fit(self, X, y, eval_set=None):
        """Fit a gradient boosting procedure to a dataset.

        Args:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                The restrictions on the data depend on the weak learner.
            y (array-like of shape (n_samples,)): Target values (strings or integers in
                classification, real numbers in regression).
            eval_set (tuple of length 2, optional, default=None): The evaluation set is a tuple
                ``(X_val, y_val)``. It has to respect the same conventions as ``X`` and ``y``.
        Returns:
            self
        """
        X = np.atleast_2d(X)
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.estimators_ = []
        self.line_searchers_ = []
        self.columns_ = []
        self.eval_scores_ = [] if eval_set else None

        # Use init_estimator for the first fit
        self.init_estimator = self.init_estimator.fit(X, y)
        y_pred = self.init_estimator.predict(X)

        # We keep training weak learners until we reach n_estimators or early stopping occurs
        for _ in range(self.n_estimators):

            # Compute the gradients of the loss for the current prediction
            gradients = self.loss.gradient(y, self._transform_y_pred(y_pred))

            # If row_sampling is lower than 1 then we're doing stochastic gradient descent
            if self.row_sampling < 1:
                n_rows = int(X.shape[0] * self.row_sampling)
                rows = self.rng.choice(X.shape[0], n_rows, replace=False)
            else:
                rows = None

            # If col_sampling is lower than 1 then we only use a subset of the features
            if self.col_sampling < 1:
                n_cols = int(X.shape[1] * self.col_sampling)
                cols = self.rng.choice(X.shape[1], n_cols, replace=False)
            else:
                cols = None

            # Subset X
            X_fit = X
            if rows is not None:
                X_fit = X_fit[rows, :]
            if cols is not None:
                X_fit = X_fit[:, cols]

            # Train a base model to fit the negative gradients
            estimators = []
            line_searchers = []

            for i, gradient in enumerate(gradients.T):

                # Fit a weak learner to the negative gradient of the loss function
                estimator = base.clone(self.base_estimator)
                estimator = estimator.fit(X_fit, -gradient if rows is None else -gradient[rows])
                estimators.append(estimator)

                # Estimate the descent direction using the weak learner
                direction = estimator.predict(X if cols is None else X[:, cols])

                # Apply line search if a line searcher has been provided
                if self.line_searcher:
                    line_searcher = copy.copy(self.line_searcher)
                    line_searcher = line_searcher.fit(y[:, i], y_pred[:, i], gradient, direction)
                    line_searchers.append(line_searcher)
                    direction = line_searcher.update(direction)

                # Move the predictions along the estimated descent direction
                y_pred[:, i] += self.learning_rate * direction

            # Store the estimator and the step
            self.estimators_.append(estimators)
            if line_searchers:
                self.line_searchers_.append(line_searchers)
            self.columns_.append(cols)

            # We're now at the end of a round so we can evaluate the model on the validation set
            if not eval_set:
                continue
            X_val, y_val = eval_set
            self.eval_scores_.append(self.eval_metric(y_val, self.predict(X_val)))

            # Check for early stopping
            if self.early_stopping_rounds and len(self.eval_scores_) > self.early_stopping_rounds:
                if self.eval_scores_[-1] > self.eval_scores_[-(self.early_stopping_rounds+1)]:
                    break

        return self

    def iter_predict(self, X, include_init=False):
        """Returns the predictions for ``X`` at every stage of the boosting procedure.

        Args:
            X (array-like or sparse matrix of shape (n_samples, n_features): The input samples.
                The restrictions on the data depend on the weak learner.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.
        Returns:
            iterator of arrays of shape (n_samples,) containing the predicted values at each stage
        """
        y_pred = self.init_estimator.predict(X)

        # The user decides if the initial prediction should be included or not
        if include_init:
            yield y_pred

        for estimators, line_searchers, cols in itertools.zip_longest(self.estimators_,
                                                                      self.line_searchers_,
                                                                      self.columns_):

            for i, (estimator, line_searcher) in enumerate(itertools.zip_longest(estimators,
                                                                                 line_searchers or [])):

                # If we used column sampling then we have to make sure the columns of X are arranged
                # in the correct order
                if cols is None:
                    direction = estimator.predict(X)
                else:
                    direction = estimator.predict(X[:, cols])

                if line_searcher:
                    direction = line_searcher.update(direction)

                y_pred[:, i] += self.learning_rate * direction

            yield y_pred

    def predict(self, X):
        """Returns the predictions for ``X``.

        Under the hood this method simply goes through the outputs of ``iter_predict`` and returns
        the final one.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                The restrictions on the data depend on the weak learner.

        Returns:
            array of shape (n_samples,) containing the predicted values.
        """
        y_preds = collections.deque(self.iter_predict(X), maxlen=1)
        return y_preds.pop()


class BoostingRegressor(BaseBoosting, base.RegressorMixin):
    """Gradient boosting for regression.

    Arguments:
        loss (starboost.losses.Loss, default=starboost.loss.L2Loss)
            The loss function that will be optimized. At every stage a weak learner will be fit to
            the negative gradient of the loss. The provided value must be a class that at the very
            least implements a ``__call__`` method and a ``gradient`` method.
        base_estimator (sklearn.base.RegressorMixin, default=None): The weak learner. This must be
            a regression model, even when using ``BoostingClassifier``.
        tree_flavor (bool, default=False): Indicates if the provided ``base_estimator`` is a tree
            model or not. Various boosting optimizations specific to trees can be made to improve
            the overall performance.
        n_estimators (int, default=30): The maximum number of weak learners to train. The final
            number of trained weak learners will be lower than ``n_estimators`` if early stopping
            happens.
        init_estimator (sklearn.base.BaseEstimator, default=None): The estimator used to make the
            initial guess. If ``None`` then the ``init_estimator`` property from the ``loss`` will
            be used.
        line_searcher (starboost.line_searchers.LineSearcher, default=None): A line searcher which
            can be used to find the optimal step size during gradient descent. If you've set
            ``tree_flavor`` to ``True`` and are using one of StarBoost's losses then an optimal
            line searcher will be used, meaning you safely set this field to ``None``.
        learning_rate (float, default=0.1): The learning rate shrinks the contribution of each tree.
            Specifically the descent direction estimated by each weak learner will be multiplied by
            ``learning_rate``. There is a trade-off between learning_rate and ``n_estimators``.
        row_sampling (float, default=1.0): The ratio of rows to sample at each stage.
        col_sampling (float, default=1.0): The ratio of columns to sample at each stage.
        eval_metric (function, default=None): The evaluation metric used to check for early
            stopping. If ``None`` it will default to ``loss``.
        random_state (int, RandomState instance or None, default=None): If int, ``random_state`` is
            the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; If ``None``, the random number
            generator is the ``RandomState`` instance used by ``np.random``.
    """

    def _transform_y_pred(self, y_pred):
        return y_pred

    @property
    def _default_loss(self):
        return losses.L2Loss()

    def iter_predict(self, X, include_init=False):
        for y_pred in super().iter_predict(X, include_init=include_init):
            yield y_pred[:, 0]


def softmax(x):
    """Can be replaced once scipy 1.3 is released."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:, None]


class BoostingClassifier(BaseBoosting, base.ClassifierMixin):
    """Gradient boosting for regression.

    Arguments:
        loss (starboost.losses.Loss, default=starboost.loss.L2Loss)
            The loss function that will be optimized. At every stage a weak learner will be fit to
            the negative gradient of the loss. The provided value must be a class that at the very
            least implements a ``__call__`` method and a ``gradient`` method.
        base_estimator (sklearn.base.RegressorMixin, default=None): The weak learner. This must be
            a regression model, even when using ``BoostingClassifier``.
        tree_flavor (bool, default=False): Indicates if the provided ``base_estimator`` is a tree
            model or not. Various boosting optimizations specific to trees can be made to improve
            the overall performance.
        n_estimators (int, default=30): The maximum number of weak learners to train. The final
            number of trained weak learners will be lower than ``n_estimators`` if early stopping
            happens.
        init_estimator (sklearn.base.BaseEstimator, default=None): The estimator used to make the
            initial guess. If ``None`` then the ``init_estimator`` property from the ``loss`` will
            be used.
        line_searcher (starboost.line_searchers.LineSearcher, default=None): A line searcher which
            can be used to find the optimal step size during gradient descent. If you've set
            ``tree_flavor`` to ``True`` and are using one of StarBoost's losses then an optimal
            line searcher will be used, meaning you safely set this field to ``None``.
        learning_rate (float, default=0.1): The learning rate shrinks the contribution of each tree.
            Specifically the descent direction estimated by each weak learner will be multiplied by
            ``learning_rate``. There is a trade-off between learning_rate and ``n_estimators``.
        row_sampling (float, default=1.0): The ratio of rows to sample at each stage.
        col_sampling (float, default=1.0): The ratio of columns to sample at each stage.
        eval_metric (function, default=None): The evaluation metric used to check for early
            stopping. If ``None`` it will default to ``loss``.
        random_state (int, RandomState instance or None, default=None): If int, ``random_state`` is
            the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; If ``None``, the random number
            generator is the ``RandomState`` instance used by ``np.random``.
    """

    def _transform_y_pred(self, y_pred):
        if self.n_classes_ > 2:
            return softmax(y_pred)
        return sigmoid(y_pred)

    @property
    def _default_loss(self):
        return losses.LogLoss()

    def fit(self, X, y, eval_set=None):
        binarizer = preprocessing.LabelBinarizer(sparse_output=False).fit(y)
        self.n_classes_ = len(binarizer.classes_)
        return super().fit(X=X, y=binarizer.transform(y), eval_set=eval_set)

    def iter_predict_proba(self, X, include_init=False):
        """Returns the predicted probabilities for ``X`` at every stage of the boosting procedure.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                The restrictions on the data depend on the weak learner.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.

        Returns:
            iterator of arrays of shape (n_samples, n_classes) containing the predicted
            probabilities at each stage
        """
        probas = np.empty(shape=(len(X), self.n_classes_), dtype=np.float32)

        for y_pred in super().iter_predict(X, include_init=include_init):
            if self.n_classes_ == 2:
                probas[:, 1] = self._transform_y_pred(y_pred[:, 0])
                probas[:, 0] = 1 - probas[:, 1]
            else:
                probas = self._transform_y_pred(y_pred)
            yield probas

    def iter_predict(self, X, include_init=False):
        """Returns the predicted classes for ``X`` at every stage of the boosting procedure.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                The restrictions on the data depend on the weak learner.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.

        Returns:
            iterator of arrays of shape (n_samples, n_classes) containing the predicted classes at
            each stage.
        """
        for probas in self.iter_predict_proba(X, include_init=include_init):
            yield np.argmax(probas, axis=1)

    def predict_proba(self, X):
        """Returns the predicted probabilities for ``X``.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                The restrictions on the data depend on the weak learner.

        Returns:
            array of shape (n_samples, n_classes) containing the predicted probabilities.
        """
        probas = collections.deque(self.iter_predict_proba(X), maxlen=1)
        return probas.pop()
