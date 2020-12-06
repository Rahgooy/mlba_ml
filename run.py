import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
from mlp import MLP
from mlba_nn import MLBA_NN
import json
from pathlib import Path
from multiprocessing import Pool
import os
import pickle
import torch


class DummyScaler:
    def transform(self, x):
        return x

    def fit_transform(self, x):
        return x


class CustomScaler:
    def __init__(self):
        self.norm = Normalizer(norm='max')
        self.st = StandardScaler()

    def transform(self, x):
        x = self.norm.transform(x)
        return self.st.transform(x)

    def fit_transform(self, x):
        x = self.norm.fit_transform(x)
        return self.st.fit_transform(x)


def load_paper_data(path):
    cols = ['Cond', 'Time', 'Resp.O1', 'Resp.O2', 'Resp.O3']
    with open(path) as f:
        data = json.load(f)
    data = [[s[k] for k in cols] for u in data for s in u]
    data = pd.DataFrame(data=data, columns=cols)
    return data.rename(columns={'Cond': 'Effect'})


def split(X, y, test_size, scaler):
    if test_size:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y.reshape(-1, 1), test_size=test_size)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        return [scaler, X_train, y_train, X_val, y_val]

    return [scaler, scaler.fit_transform(X), y]


features = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
epochs = 10
models = {
    'mlp_crim': {
        'data': 'Criminals',
        'model': lambda: MLP(6, 3, 50, epochs, 1024, 0.001),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0, CustomScaler()),
    },
    'mlp_rect': {
        'data': 'Rectangles',
        'model': lambda: MLP(6, 3, 50, epochs, 32, 0.001),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0, DummyScaler()),
    },
    'mlba_nn_crim': {
        'data': 'Criminals',
        'model': lambda: MLBA_NN(6, 3, 50, epochs, 1024, 0.0005, weight_decay=0.1, dropout=0),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.33, CustomScaler()) + [True],
    },
    'mlba_nn_rect': {
        'data': 'Rectangles',
        'model': lambda: MLBA_NN(6, 3, 50, epochs, 1024, 0.001, weight_decay=0.1, dropout=0),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.33, CustomScaler()) + [True],
    },
}

experimentData = {
    'Rectangles': {
        'name': 'Rectangles',
        'train': pd.read_csv('data/E2.csv'),
        'test': {'e1a': pd.read_csv('data/E1a.csv'),
                 'e1b': pd.read_csv('data/E1b.csv'),
                 'e1c': pd.read_csv('data/E1c.csv')},
        'paper': {'e1a': load_paper_data('hb_mlba/e1a.pred.json'),
                  'e1b': load_paper_data('hb_mlba/e1b.pred.json'),
                  'e1c': load_paper_data('hb_mlba/e1c.pred.json'), }
    },
    'Criminals': {
        'name': 'Criminals',
        'train': pd.read_csv('data/E4.csv'),
        'test': {'e3a': pd.read_csv('data/E3a.csv'),
                 'e3b': pd.read_csv('data/E3b.csv'),
                 'e3c': pd.read_csv('data/E3c.csv')},
        'paper': {'e3a': load_paper_data('hb_mlba/e3a.pred.json'),
                  'e3b': load_paper_data('hb_mlba/e3b.pred.json'),
                  'e3c': load_paper_data('hb_mlba/e3c.pred.json'), }
    }
}


def get_freq(x):
    hist, _ = np.histogram(x, 3)
    return hist/hist.sum()


def predict(model, data, scaler):
    X_test = scaler.transform(data[features].values)
    y_test = data.response.values - 1
    actual_freq = get_freq(y_test)
    pred_freq = model.predict_proba(X_test).mean(0)
    return actual_freq, pred_freq, ((actual_freq - pred_freq) ** 2).mean()


def get_predictions(model, data, scaler):
    effects = data.groupby('Effect')
    effects_data = [d for _, d in effects]
    MSE = []
    actual = []
    pred = []
    for d in effects_data:
        actual_freq, pred_freq, mse = predict(model, d, scaler)
        MSE.append(mse)
        actual.append(actual_freq)
        pred.append(pred_freq)

    actual = np.array(actual)
    pred = np.array(pred)
    MSE = np.array(MSE)
    return actual, pred, MSE, [g for g, _ in effects]


markers = ['o', '^', 'd', 's', '.', '*', 'x', 'p', 'h', 'v']
colors = ['r', 'lime', 'b']

def save_results(path, actual_list, pred_list, mse_list, names, paper_pred):
    mse = sum(mse_list) / len(mse_list)
    with path.with_name(path.name + '_mse.txt').open(mode='w') as f:
        for i in range(len(names)):
            f.write(f'{names[i]}: {mse[i]}\n')

    with path.with_name(path.name + '_preds.pkl').open(mode='wb') as f:
        pickle.dump((names, actual_list, pred_list, paper_pred), f)


def save_model(model, scaler, model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open('bw') as f:
        pickle.dump((model, scaler), f)


def load_model(model_path):
    with model_path.open('br') as f:
        (model, scaler) = pickle.load(f)
    return model, scaler


def run_model(m, exp, run):
    print(f'[{os.getpid()}] Run #{run} ...')
    np.random.seed(os.getpid() * 100 + run)
    X_train = exp['train'][features].values
    y_train = (exp['train'].response.values - 1)

    model = models[m]['model']()
    params = models[m]['params'](X_train, y_train)
    scaler = params[0]
    params = params[1:]
    model.fit(*params)
    models_path = Path(f'out/temp/models')
    save_model(model, scaler, models_path / f'{m}_run_{run}.pkl')
    (temp_m, temp_scaler) = load_model(models_path / f'{m}_run_{run}.pkl')
    actual = {}
    pred = {}
    mse = {}
    names = {}
    for e in exp['test']:
        data = exp['test'][e]
        a, p, m, effects = get_predictions(model, data, scaler)
        actual[e] = a
        mse[e] = m
        pred[e] = p
        names[e] = effects

    return mse, actual, pred, names


def evaluate(m, n=10, jobs=5):
    exp = experimentData[models[m]['data']]
    print(f'[{os.getpid()}] Evaluating {m} on {exp["name"]} data...')
    dir = Path(f'out/res/{exp["name"]}/{m}/')
    dir.mkdir(parents=True, exist_ok=True)

    args = [(m, exp, i+1) for i in range(n)]
    with Pool(jobs) as p:
        results = p.starmap(run_model, args)

    for e in exp['test']:
        mse_list = [r[0][e] for r in results]
        actual_list = [r[1][e] for r in results]
        pred_list = [r[2][e] for r in results]
        names = results[0][3][e]
        paper_data = exp['paper'][e]
        p_pred = np.array([paper_data[paper_data.Effect == eff][[
                          'Resp.O1', 'Resp.O2', 'Resp.O3']].values.mean(0) for eff in names])

        save_results(dir / f'{m}_{e}', actual_list,
                     pred_list, mse_list, names, p_pred)


def run():
    for m in models:
        evaluate(m, n=5, jobs=5)


if __name__ == "__main__":
    run()
