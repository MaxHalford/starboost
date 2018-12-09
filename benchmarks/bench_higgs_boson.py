import collections
import gzip
import os
from urllib import request

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree
from sklearn.externals import joblib
import starboost as sb



HERE = './'
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
m = joblib.Memory(location='/tmp', mmap_mode='r')

MAX_DEPTH = 3
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
SUBSAMPLE = 50000


@m.cache
def load_data():
    filename = os.path.join(HERE, URL.rsplit('/', 1)[-1])
    if not os.path.exists(filename):
        print(f'Downloading {URL} to {filename}...')
        request.urlretrieve(URL, filename)
    print(f'Parsing {filename}...')
    with gzip.GzipFile(filename) as f:
        df = pd.read_csv(f, header=None, dtype=np.float32)
    return df


df = load_data()

if SUBSAMPLE:
    df = df[:SUBSAMPLE]

y = df.values[:, 0]
X = df.values[:, 1:]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)


val_scores = collections.defaultdict(list)
metric = metrics.mean_squared_error

model = sb.BoostingRegressor(
    loss=sb.losses.L2Loss(),
    base_estimator=tree.DecisionTreeRegressor(max_depth=MAX_DEPTH, presort=True),
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    row_sampling=1.0,
    col_sampling=1.0,
    random_state=42
)
model.fit(X_train, y_train);
for y_pred in model.iter_predict(X_test):
    val_scores['StarBoost'].append(metric(y_test, y_pred))


model = ensemble.GradientBoostingRegressor(
    max_depth=MAX_DEPTH,
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    random_state=42
)

model.fit(X_train, y_train);
for y_pred in model.staged_predict(X_test):
    val_scores['scikit-learn'].append(metric(y_test, y_pred))


model = lgbm.LGBMRegressor(
    max_depth=MAX_DEPTH,
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    random_state=42
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), eval_names=['test'], verbose=False);
for score in model.evals_result_['test']['l2']:
    val_scores['LightGBM'].append(score)

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(9, 6))

for name, scores in val_scores.items():
    ax.plot(scores, label=name, alpha=0.75)
ax.legend()
