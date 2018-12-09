import abc

import numpy as np

from . import init
from . import line_searchers


class Loss(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def gradient(self, y_true, y_pred):
        pass

    @property
    @abc.abstractmethod
    def default_init_estimator(self):
        pass

    @property
    def tree_line_searcher(self):
        return None


class L2Loss(Loss):
    """Computes the L2 loss, also known as the mean squared error.

    Mathematically, the L2 loss is defined as

    :math:`L = \\frac{1}{n} \\sum_i^n (p_i - y_i)^2`

    It's gradient is

    :math:`\\frac{\\partial L}{\\partial y_i} = p_i`

    Using `MSE` is equivalent to setting the `loss` parameter to `ls` in scikit-learn's
    `GradientBoostingRegressor.`
    """

    def __call__(self, y_true, y_pred):
        """Returns the L2 loss.

        Example:
            >>> import starboost as sb
            >>> y_true = [10, 25, 0]
            >>> y_pred = [5, 30, 5]
            >>> sb.losses.L2Loss()(y_true, y_pred)
            25.0
        """
        return np.power(np.subtract(y_pred, y_true), 2).mean()

    def gradient(self, y_true, y_pred):
        """Returns the gradient of the L2 loss with respect to each prediction.

        Example:
            >>> import starboost as sb
            >>> y_true = [10, 25, 0]
            >>> y_pred = [5, 30, 5]
            >>> sb.losses.L2Loss().gradient(y_true, y_pred)
            array([-5,  5,  5])
        """
        return np.subtract(y_pred, y_true)

    @property
    def default_init_estimator(self):
        """Returns ``starboost.init.MeanEstimator()``."""
        return init.MeanEstimator()

    @property
    def tree_line_searcher(self):
        return None


class L1Loss(Loss):
    """Computes the L1 loss, also known as the mean absolute error.

    Mathematically, the L1 loss is defined as

    :math:`L = \\frac{1}{n} \\sum_i^n |p_i - y_i|`

    It's gradient is

    :math:`\\frac{\\partial L}{\\partial y_i} = sign(p_i - y_i)`

    where :math:`sign(p_i - y_i)` is equal to 0 if :math:`p_i` is equal to :math:`y_i`. Note that
    this is slightly different from scikit-learn, which replaces 0s by -1s.

    Using ``L1Loss`` produces mostly the same results as when setting the ``loss`` parameter to
    ``'lad'`` in scikit-learn's ``GradientBoostingRegressor``.
    """

    def __call__(self, y_true, y_pred):
        """Returns the L1 loss.

        Example:
            >>> import starboost as sb
            >>> y_true = [0, 0, 1]
            >>> y_pred = [0.5, 0.5, 0.5]
            >>> sb.losses.L1Loss()(y_true, y_pred)
            0.5
        """
        return np.abs(np.subtract(y_pred, y_true)).mean()

    def gradient(self, y_true, y_pred):
        """Returns the gradient of the L1 loss with respect to each prediction.

        Example:
            >>> import starboost as sb
            >>> y_true = [0, 0, 1]
            >>> y_pred = [0.3, 0, 0.8]
            >>> sb.losses.L1Loss().gradient(y_true, y_pred)
            array([ 1.,  0., -1.])
        """
        return np.sign(np.subtract(y_pred, y_true))

    @property
    def default_init_estimator(self):
        """Returns ``starboost.init.QuantileEstimator(alpha=0.5)``."""
        return init.QuantileEstimator(alpha=0.5)

    @property
    def tree_line_searcher(self):
        """When using ``L1Loss`` the gradient descent procedure will chase the negative of
        ``L1Loss``'s gradient. The negative of the gradient is solely composed of 1s, -1s, and 0s.
        It turns out that replacing the estimated descent direction with the median of the
        according residuals will in fact minimize the overall mean absolute error much faster.

        This is exactly the same procedure scikit-learn uses to modify the leaves of decision trees
        in ``GradientBoostingRegressor``. However this procedure is more generic and works with any
        kind of weak learner.
        """
        def update_leaf(y_true, y_pred, gradient):
            return np.median(y_true - y_pred)

        return line_searchers.LeafLineSearcher(update_leaf=update_leaf)


class LogLoss(Loss):
    """Computes the logarithmic loss.

    Mathematically, the L1 loss is defined as

    :math:`L = -\\frac{1}{n} \\sum_i^n y_i log(p_i) + (1-y_i) log(1-p_i)`

    It's gradient is

    :math:`\\frac{\\partial L}{\\partial y_i} = sign(p_i - y_i)`

    This loss works for binary classification as well as for multi-class cases (in which case the
    loss is usually referred to as "cross-entropy").
    """

    def __call__(self, y_true, y_pred):
        """Returns the log loss.

        Example:
            >>> import starboost as sb
            >>> y_true = [0, 0, 1]
            >>> y_pred = [0.5, 0.5, 0.5]
            >>> sb.losses.LogLoss()(y_true, y_pred)
            0.807410...
        """
        loss = -((np.multiply(y_true, y_pred)) - np.logaddexp(0., y_pred))
        return loss.mean()

    def gradient(self, y_true, y_pred):
        """Returns the gradient of the log loss with respect to each prediction.

        Example:
            >>> import starboost as sb
            >>> y_true = [0, 0, 1]
            >>> y_pred = [0.5, 0.5, 0.5]
            >>> sb.losses.LogLoss().gradient(y_true, y_pred)
            array([ 0.5,  0.5, -0.5])
        """
        return np.subtract(y_pred, y_true)

    @property
    def default_init_estimator(self):
        """Returns ``starboost.init.PriorProbabilityEstimator()``."""
        return init.PriorProbabilityEstimator()

    @property
    def tree_line_searcher(self):

        def update_leaf(y_true, y_pred, gradient):
            numerator = np.sum(-gradient)
            denominator = np.sum((y_true + gradient) * (1 - y_true - gradient))
            return (numerator / denominator) if denominator > 1e-150 else 0.

        return line_searchers.LeafLineSearcher(update_leaf=update_leaf)
