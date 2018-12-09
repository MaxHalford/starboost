from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree

import starboost as sb

X, y = datasets.load_breast_cancer(return_X_y=True)

X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

def micro_f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')

model = sb.BoostingClassifier(
    loss=sb.losses.LogLoss(),
    base_estimator=tree.DecisionTreeRegressor(max_depth=3, presort=True),
    tree_flavor=True,
    n_estimators=30,
    init_estimator=sb.init.PriorProbabilityEstimator(),
    learning_rate=0.1,
    row_sampling=0.8,
    col_sampling=0.8,
    eval_metric=micro_f1_score,
    early_stopping_rounds=5,
    random_state=42
)

model = model.fit(X_fit, y_fit, eval_set=(X_val, y_val))

y_pred = model.predict_proba(X_val)

print(metrics.roc_auc_score(y_val, y_pred[:, 1]))
