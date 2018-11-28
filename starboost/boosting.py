import abc
import collections
import warnings

import numpy as np
from scipy import optimize
from scipy.special import expit as sigmoid
from sklearn import base
from sklearn import preprocessing
from sklearn import utils


warnings.simplefilter('ignore', np.RankWarning)


class BaseBoosting(abc.ABC, base.BaseEstimator):
    """Gradient Boosting for classification.

    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    """

    def __init__(self, loss, base_estimator, n_estimators, init_estimator=None, learning_rate=0.1,
                 row_sampling=1.0, col_sampling=1.0, eval_metric=None, early_stopping_rounds=None,
                 random_state=None):
        self.loss = loss
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.init_estimator = init_estimator or self.loss.init_estimator
        self.learning_rate = learning_rate
        self.row_sampling = row_sampling
        self.col_sampling = col_sampling
        self.eval_metric = eval_metric or loss
        self.early_stopping_rounds = early_stopping_rounds
        self.rng = utils.check_random_state(random_state)

    @abc.abstractmethod
    def transform_y_pred(self, y_pred):
        pass

    def fit(self, X, y, eval_set=None):
        """Fit a gradient boosting procedure to a dataset.

        """

        self.estimators_ = []
        self.steps_ = []
        self.alterations_ = []
        self.columns_ = []
        self.eval_scores_ = [] if eval_set else None

        # Use init_estimator for the first fit
        self.init_estimator = self.init_estimator.fit(X, y)
        y_pred = self.init_estimator.predict(X)

        # We keep training weak learners until we reach n_estimators or early stopping occurs
        for _ in range(self.n_estimators):

            # Compute the gradients of the loss for the current prediction
            gradient = self.loss.gradient(y, self.transform_y_pred(y_pred))

            X_fit = X

            # If row_sampling is lower than 1 then we're doing stochastic gradient descent
            if self.row_sampling < 1:
                n_rows = int(X.shape[0] * self.row_sampling)
                rows = self.rng.choice(X.shape[0], n_rows, replace=False)
                X_fit = X[rows, :]
                gradient = gradient[rows]

            # If col_sampling is lower than 1 then we only use a subset of the features
            if self.col_sampling < 1:
                n_cols = int(X.shape[1] * self.col_sampling)
                cols = self.rng.choice(X.shape[1], n_cols, replace=False)
                X_fit = X[:, cols]
                # We have to memorise which columns we used so that we can produce the correct
                # predictions at test time
                self.columns_.append(cols)
            else:
                self.columns_.append(None)

            # Train a base model to fit the negative gradients
            estimator = base.clone(self.base_estimator)
            estimator = estimator.fit(X_fit, -gradient)
            direction = estimator.predict(X)

            # Depending on the loss the direction might need to be corrected,
            # which is the case for MSE but not for MAE
            if hasattr(self.loss, 'alter_direction'):
                new_direction = self.loss.alter_direction(direction, y, y_pred, gradient)
                # The thing is that we also need to be able to alter the directions at test time.
                # Gradient boosting implementations usually handle this by tinkering with the base
                # learners. This works fine with trees because the leaves can be modified in place;
                # however this doesn't necessarily work with other models. The trick is to fit a
                # polynomial between the old direction and the new one. We can then store polynomial
                # and use them at test time to alter the directions.
                unique, indices = np.unique(direction, return_index=True)
                coefs = np.polyfit(unique, new_direction[indices], deg=len(unique)-1)
                poly = np.poly1d(coefs)
                self.alterations_.append(poly)
                direction = new_direction
            else:
                self.alterations_.append(None)

            # Perform line search to find the step size
            step, *_ = optimize.line_search(
                f=lambda x: self.loss(y, self.transform_y_pred(x)),
                myfprime=lambda x: self.loss.gradient(y, self.transform_y_pred(x)),
                xk=y_pred,
                pk=direction
            )

            # If the line search didn't converge then we use 1
            step = step or 1

            # We now move y_pred along the descent direction using shrinkage
            y_pred += self.learning_rate * step * direction

            # Store the estimator and the step
            self.estimators_.append(estimator)
            self.steps_.append(step)

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
        """Returns the predictions for ``X`` at every stage of the boosting procedure."""

        y_pred = self.init_estimator.predict(X)

        # The user decides if the initial prediction should be included or not
        if include_init:
            yield y_pred

        for step, estimator, alteration, cols in zip(self.steps_, self.estimators_,
                                                     self.alterations_, self.columns_):

            # If we used column sampling then we have to make sure the columns of X are arranged
            # in the correct order
            if cols:
                direction = estimator.predict(X[:, cols])
            else:
                direction = estimator.predict(X)

            if alteration:
                direction = alteration(direction)

            y_pred += step * self.learning_rate * direction
            yield y_pred

    def predict(self, X):
        """Returns the predictions for ``X``.

        Under the hood this method simply goes through the outputs of ``iter_predict`` and returns
        the final one.
        """
        y_preds = collections.deque(self.iter_predict(X), maxlen=1)
        return y_preds.pop()


class BoostingRegressor(BaseBoosting, base.RegressorMixin):

    def transform_y_pred(self, y_pred):
        return y_pred

    def fit(self, X, y):
        return super().fit(X, y)


class BoostingClassifier(BaseBoosting, base.ClassifierMixin):

    def transform_y_pred(self, y_pred):
        return sigmoid(y_pred)

    # def fit(self, X, y):
    #     self.lb_ = preprocessing.LabelBinarizer(sparse_output=False).fit(y)
    #     return super().fit(X, self.lb_.transform(y))

    def iter_predict_proba(self, X, include_init=False):

        probas = np.empty(shape=(len(X), 2), dtype=np.float32)

        for y_pred in super().iter_predict(X, include_init=include_init):
            probas[:, 1] = sigmoid(y_pred)
            probas[:, 0] = 1 - probas[:, 1]
            yield probas

    def iter_predict(self, X, include_init=False):
        for probas in self.iter_predict_proba(X, include_init=include_init):
            yield np.argmax(probas, axis=1)

    def predict_proba(self, X):
        probas = collections.deque(self.iter_predict_proba(X), maxlen=1)
        return probas.pop()
