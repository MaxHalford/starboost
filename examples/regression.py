from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree

import starboost as sb

X, y = datasets.load_boston(return_X_y=True)

X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5

model = sb.BoostingRegressor(
    loss=sb.losses.L2Loss(),
    base_estimator=tree.DecisionTreeRegressor(max_depth=3, presort=True),
    tree_flavor=True,
    n_estimators=30,
    init_estimator=linear_model.LinearRegression(),
    learning_rate=0.1,
    row_sampling=0.8,
    col_sampling=0.8,
    eval_metric=rmse,
    early_stopping_rounds=5,
    random_state=42
)

model = model.fit(X_fit, y_fit, eval_set=(X_val, y_val))

y_pred = model.predict(X_val)

print(rmse(y_val, y_pred))
