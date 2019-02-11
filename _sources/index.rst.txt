.. image:: https://docs.google.com/drawings/d/e/2PACX-1vQKggEpm0PGgmkB7LymYmHdptSFEwYXC5yecuph_0gGmZ5fW-bTIfowcDLHVHxjgKQTHq8Y21H0d5LF/pub?w=1277&h=375
   :height: 120px
   :width: 400px
   :alt: logo
   :align: center

.. toctree::
   :maxdepth: 2

.. automodule:: starboost

**What is this?**

This is the documentation for StarBoost, a Python library that implements gradient boosting. Gradient boosting is an efficient and popular machine learning algorithm used for supervised learning.

**Doesn't scikit-learn already do that?**

Indeed scikit-learn `implements gradient boosting <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_, but the only supported weak learner is decision tree. In essence gradient boosting can be used with other weak learners than decision trees.

**What about XGBoost/LightGBM/CatBoost?**

The mentioned libraries are the state of the art of gradient boosting decision trees (GBRT). They implement a specific version of gradient boosting that is tailored to decision trees. StarBoost's purpose isn't to compete with them. Instead it's goal is to implement a generic gradient boosting algorithm that works with any weak learner.

A focus of StarBoost is to keep the code readable and commented, instead of obfuscating the algorithm under a pile of tangled code.

**What's a weak learner?**

A weak learner is any machine learning model that can learn from labeled data. It's called "weak" because it usually works better as part of an ensemble (such as gradient boosting). Examples are linear models, radial basis functions, decision trees, genetic programming, neural networks, etc. In theory you could even use gradient boosting as a weak learner.

**Is it compatible with scikit-learn?**

Yes, it is.

**How do I install it?**

Barring any weird Python setup, you simply have to run ``pip install starboost``.

**How do I use it?**

The following snippet shows a very basic usage of StarBoost. Please check out the `examples directory on GitHub <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_ for comprehensive examples.

.. code-block:: python

    from sklearn import datasets
    from sklearn import tree
    import starboost as sb

    X, y = datasets.load_boston(return_X_y=True)

    model = sb.BoostingRegressor(
        base_estimator=tree.DecisionTreeRegressor(max_depth=3),
        n_estimators=30,
        learning_rate=0.1
    )

    model = model.fit(X, y)

    y_pred = model.predict(X)

**What are you planning on doing next?**

- Logging the progress
- Handling sample weights
- Implement more loss functions
- Make it faster
- Newton boosting (taking into account the information from the Hessian)
- Learning to rank

**By the way, why is it called "StarBoost"?**

As you might already know, in programming the star symbol ``*`` often refers to the concept of "everything". The idea is that StarBoost can be used with any weak learner, not just decision trees.

.. _api:

Boosting
========

Regression
----------

.. autoclass:: BoostingRegressor
    :members: fit, predict, iter_predict

Classification
--------------

.. autoclass:: BoostingClassifier
    :members: fit, predict, predict_proba, iter_predict, iter_predict_proba

Losses
======

L1 loss
-------

.. autoclass:: starboost.losses.L1Loss
    :members:

    .. automethod:: __call__

L2 loss
-------

.. autoclass:: starboost.losses.L2Loss
    :members:

    .. automethod:: __call__

Log loss
--------

.. autoclass:: starboost.losses.LogLoss
    :members:

    .. automethod:: __call__

Line searchers
==============

During gradient descent the negative gradient of the loss function indicates the direction of descent. A line searcher can be used to determine how far to pursue the direction, or in other words the step size.

Line search per leaf
--------------------

One of the reasons why gradient boosting is often used with decision trees is that optimal step sizes exist and are easy to compute.

.. autoclass:: starboost.line_searchers.LeafLineSearcher
    :members:

    .. automethod:: __call__
