import abc

import numpy as np


class LineSearcher(abc.ABC):

    @abc.abstractmethod
    def fit(self, y_true, y_pred, gradient, direction):
        pass

    @abc.abstractmethod
    def update(self, direction):
        pass


class LeafLineSearcher(LineSearcher):

    def __init__(self, update_leaf):
        self.update_leaf = update_leaf

    def fit(self, y_true, y_pred, gradient, direction):
        leaves = np.unique(direction)
        self.updates_ = {}

        for leaf in leaves:
            mask = direction == leaf
            self.updates_[leaf] = self.update_leaf(y_true[mask], y_pred[mask], gradient[mask])

        return self

    def update(self, direction):

        for leaf, update in self.updates_.items():
            np.place(direction, direction == leaf, update)

        return direction
