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


def get_param_permutations(params, idx, result):
    h = tuple([(k, params[k][idx[k]]) for k in idx])
    if h in result:
        return
    result.add(h)
    for k in params:
        if idx[k] < len(params[k]) - 1:
            idx[k] += 1
            get_param_permutations(params, idx, result)
            idx[k] -= 1


def eval_fold(m, c, c_id, train_index, test_index, fold):
    model = models[m]['estimator']
    mse = 0
    for i in range(5):
        print(
            '\n' + f'[{os.getpid()}]=== Config #{c_id} Fold #{fold} try #{i+1} - {c}')
        config = {k: v for k, v in c}
        np.random.seed(os.getpid() * 100 + i)
        estimator = model(config)
        estimator.fit(X[train_index], y[train_index])
        pred = estimator.predict_proba(X[test_index]).mean(0)

        resp_freq, _ = np.histogram(y, 3)
        actual = resp_freq / len(y)
        mse += ((actual - pred)**2).mean()

    return mse / 5


def eval_model(m, c, c_id):
    f = 1
    mse = []
    for train_index, test_index in kf.split(X):
        mse.append(eval_fold(m, c, c_id, train_index, test_index, f))
        f += 1

    return mse


def cv_model(m, jobs):
    params = models[m]['params']
    idx = {k: 0 for k in params}

    permutations = set()
    get_param_permutations(params, idx, permutations)
    permutations = sorted(list(permutations))
    args = [(m, p, i+1) for i, p in enumerate(permutations)]
    evals = {}
    with Pool(jobs) as p:
        results = p.starmap(eval_model, args)

    for h, r in zip(permutations, results):
        evals[h] = r

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


cv_model(list(models.keys())[-1], 5)
