.. StarBoost documentation master file, created by
   sphinx-quickstart on Mon Nov 26 17:22:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://docs.google.com/drawings/d/e/2PACX-1vQKggEpm0PGgmkB7LymYmHdptSFEwYXC5yecuph_0gGmZ5fW-bTIfowcDLHVHxjgKQTHq8Y21H0d5LF/pub?w=1277&h=375
   :scale: 45 %
   :alt: logo
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: starboost

**What is this?**

This is the documentation for StarBoost, a Python library that implements gradient boosting. Gradient boosting is an efficient and popular machine learning algorithm used for supervised learning.

**Doesn't scikit-learn already do that?**

Indeed scikit-learn `implements gradient boosting <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_, but the only supported weak learners are decision trees. In essence gradient boosting can be used with other weak learners than decision trees.

**What about XGBoost/LightGBM/CatBoost?**

The mentionned libraries are the state of the art of gradient boosting decision trees (GBRT). They implement a specific version of gradient boosting that is tailored to decision trees. StarBoost's purpose isn't to compete with them. Instead it's goal is to implement a generic gradient boosting algorithm that works with any weak learner.

**What's a weak learner?**

A weak learner is any machine learning model that can learn from labeled data. It's called "weak" because it usually works better as part of an ensemble (such as gradient boosting). Examples are linear models, radial basis functions, decision trees, genetic programming, neural networks, etc. In theory you could even use gradient boosting as a weak learner.

**Is it compatible with scikit-learn?**

Yes, it is.

**How do I install it?**

Barring any weird Python setup, you simply have to run ``pip install starboost``.

**How do I use it?**

The following snippet shows a very basic usage of StarBoost.

.. code-block:: python

    from sklearn import datasets
    from sklearn import tree
    import starboost as sb

    X, y = datasets.load_boston(X, return_X_y=True)

    model = sb.GradientBoostingRegressor(
        base_estimator=tree.DecisionTreeRegressor(max_depth=3)
        n_estimators=30,
        learning_rate=0.1
    )

    model = model.fit(X, y)

    y_pred = model.predict(X)

**Where can I find a list of all the possible parameters?**

Please check out the :ref:`api` section.

**Are there any benchmarks?**

Yes, TODO. There are also some unit tests that proof check the outputs.

**What are you planning on doing next?**

- Handling sample weights
- Implement more loss functions
- Newton boosting (taking into account the information from the Hessian)
- Learning to rank

**By the way, why is it called "StarBoost"?**

As you might already know, in programming the star symbol ``*`` often refers to the concept of "everything". The idea is that StarBoost can be used with any weak learner, not just decision trees.

.. _api:

===
API
===

--------
Boosting
--------

.. autoclass:: BoostingRegressor
    :members:

.. autoclass:: BoostingClassifier
    :members:

------
Losses
------

.. autoclass:: starboost.loss.L1Loss
    :members:

    .. automethod:: __call__

.. autoclass:: starboost.loss.L2Loss
    :members:

    .. automethod:: __call__
