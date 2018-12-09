<div align="center">
  <!-- Logo -->
  <img src="https://docs.google.com/drawings/d/e/2PACX-1vQKggEpm0PGgmkB7LymYmHdptSFEwYXC5yecuph_0gGmZ5fW-bTIfowcDLHVHxjgKQTHq8Y21H0d5LF/pub?w=1277&h=375" alt="logo" width=550px/>
</div>

<div align="center">
  <!-- Build status -->
  <a href="https://travis-ci.org/MaxHalford/starboost">
    <img src="https://img.shields.io/travis/MaxHalford/starboost/master.svg?style=flat-square" alt="build_status" />
  </a>
  <!-- Test coverage -->
  <a href="https://coveralls.io/github/MaxHalford/starboost?branch=master">
    <img src="https://coveralls.io/repos/github/MaxHalford/starboost/badge.svg?branch=master&style=flat-square" alt="test_coverage" />
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT">
    <img src="http://img.shields.io/:license-mit-ff69b4.svg?style=flat-square" alt="license"/>
  </a>
</div>

<br/>
<br/>

<div align="center">
Please check out the <a href="https://maxhalford.github.io/starboost/">website</a> if you're looking for the documentation!
</div>

**What is this?**

This is StarBoost, a Python library that implements gradient boosting. Gradient boosting is an efficient and popular machine learning algorithm used for supervised learning.

**Doesn't scikit-learn already do that?**

Indeed scikit-learn [implements gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), but the only supported weak learner is a decision tree. In essence gradient boosting can be used with other weak learners than decision trees.

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

The following snippet shows a very basic usage of StarBoost. Please check out the [examples directory](examples/) for comprehensive examples.

```python
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
```

You can find the source code for running the benchmarks [here](benchmarks/).

**What are you planning on doing next?**

- Logging the progress
- Handling sample weights
- Implement more loss functions
- Make it faster
- Newton boosting (taking into account the information from the Hessian)
- Learning to rank

**By the way, why is it called "StarBoost"?**

As you might already know, in programming the star symbol `*` often refers to the concept of "everything". The idea is that StarBoost can be used with any weak learner, not just decision trees.


## License

The MIT License (MIT). Please see the [LICENSE file](LICENSE.md) for more information.
