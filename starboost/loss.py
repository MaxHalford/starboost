import numpy as np

from . import init


class L2Loss:
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
            >>> sb.loss.L2Loss()(y_true, y_pred)
            25.0
        """
        return np.power(np.subtract(y_pred, y_true), 2).mean()

    def gradient(self, y_true, y_pred):
        """Returns the gradient of the L2 loss with respect to each prediction.

        Example:
            >>> import starboost as sb
            >>> y_true = [10, 25, 0]
            >>> y_pred = [5, 30, 5]
            >>> sb.loss.L2Loss().gradient(y_true, y_pred)
            array([-5,  5,  5])
        """
        return np.subtract(y_pred, y_true)

    @property
    def init_estimator(self):
        """Returns ``starboost.init.MeanEstimator()``."""
        return init.MeanEstimator()


class L1Loss:
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
            >>> sb.loss.L1Loss()(y_true, y_pred)
            0.5
        """
        return np.abs(np.subtract(y_pred, y_true)).mean()

    def gradient(self, y_true, y_pred):
        """Returns the gradient of the L1 loss with respect to each prediction.

        Example:
            >>> import starboost as sb
            >>> y_true = [0, 0, 1]
            >>> y_pred = [0.3, 0, 0.8]
            >>> sb.loss.L1Loss().gradient(y_true, y_pred)
            array([ 1.,  0., -1.])
        """
        return np.sign(np.subtract(y_pred, y_true))

    @property
    def init_estimator(self):
        """Returns ``starboost.init.QuantileEstimator(alpha=0.5)``."""
        return init.QuantileEstimator(alpha=0.5)

    def alter_direction(self, direction, y_true, y_pred, gradient):
        """When using ``L1Loss`` the gradient descent procedure will chase the negative of
        ``L1Loss``'s gradient. The negative of the gradient is solely composed of 1s, -1s, and 0s.
        It turns out that replacing the estimated descent direction with the median of the
        according residuals will in fact minimize the overall mean absolute error much faster.

        This is exactly the same procedure scikit-learn uses to modify the leaves of decision trees
        in ``GradientBoostingRegressor``. However this procedure is more generic and works with any
        kind of weak learner.
        """
        resi = y_true - y_pred
        unique = np.unique(direction)
        repl = {}
        for u in unique:
            diff = resi[direction == u]
            repl[u] = np.median(diff)
        return np.fromiter((repl[u] for u in direction), direction.dtype, count=len(direction))


class LogLoss:
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
            >>> sb.loss.LogLoss()(y_true, y_pred)
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
            >>> sb.loss.LogLoss().gradient(y_true, y_pred)
            array([ 0.5,  0.5, -0.5])
        """
        return np.subtract(y_pred, y_true)

    @property
    def init_estimator(self):
        """Returns ``starboost.init.PriorProbabilityEstimator()``."""
        return init.PriorProbabilityEstimator()

    def alter_direction(self, direction, y_true, y_pred, gradient):
        unique = np.unique(direction)
        repl = {}

        for u in unique:
            residual = -gradient[direction == u]
            y = y_true[direction == u]
            numerator = np.sum(residual)
            denominator = np.sum((y - residual) * (1 - y + residual))
            repl[u] = (numerator / denominator) if denominator > 1e-150 else 0.

        return np.fromiter((repl[u] for u in direction), direction.dtype, count=len(direction))
