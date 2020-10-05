import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlp import MLP
from mlba_nn import MLBA_NN
import json
from pathlib import Path
from multiprocessing import Pool
import os
from sklearn.model_selection import KFold, train_test_split

features = ['Rect1Height', 'Rect1Width', 'Rect2Height',
            'Rect2Width', 'Rect3Height', 'Rect3Width']
models = {
    'mlba_nn': {
        'estimator': lambda c: MLBA_NN(c['n_features'], c['n_options'], c['n_hidden'],
                                       c['n_epochs'], c['batch'], c['lr']),
        'params': {
            'n_features': [6],
            'n_options': [3],
            'n_hidden': [50, 100],
            'n_epochs': [100],
            'batch': [16, 32, 64],
            'optim': ['Adam', 'RMSProp', 'SGD'],
            'lr': [0.001, 0.01, 0.1],
        }
    },
}

data = pd.read_csv('data/E2.csv')
X = data[features].values
y = (data.response.values - 1)

# Only use a portion of the data
X, _, y, _ = train_test_split(X, y, train_size=0.3)

kf = KFold(n_splits=5)


def get_freq(x):
    hist, _ = np.histogram(x, 3)
    return hist/hist.sum()


def eval_model(model, params, idx, evals):
    h = tuple([(k, idx[k]) for k in idx])
    if h in evals:
        return
    c = {k: params[k][idx[k]] for k in params}
    mse = []
    f = 1
    for train_index, test_index in kf.split(X):
        m = 0
        for i in range(5):
            print('\n' + f'=== Config #{len(evals) + 1} Fold #{f} try #{i+1} - {c}')
            estimator = model(c)
            estimator.fit(X[train_index], y[train_index])
            pred = estimator.predict_proba(X[test_index]).mean(0)

            resp_freq, _ = np.histogram(y, 3)
            actual = resp_freq / len(y)
            m += ((actual - pred)**2).mean()
        mse.append(m/5)

        f += 1
    evals[h] = mse

    for k in params:
        if idx[k] < len(params[k]) - 1:
            idx[k] += 1
            eval_model(model, params, idx, evals)
            idx[k] -= 1


def eval_process(m):
    model = models[m]['estimator']
    params = models[m]['params']
    idx = {k: 0 for k in params}

    evals = {}
    eval_model(model, params, idx, evals)
    file = Path(f'out/cv/{m}.txt')
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w') as f:
        for e in sorted(evals.keys(), key=lambda k: sum(evals[k])):
            f.write('Mean MSE: ')
            f.write(str(sum(evals[e])/len(evals[e])))
            f.write(', Params: ')
            f.write(str(e))
            f.write(', Folds MSE: ')
            f.write(str(evals[e]))
            f.write('\n')

    return evals

eval_process(list(models.keys())[-1])
