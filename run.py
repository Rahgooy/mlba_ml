import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
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


def load_paper_data(path):
    cols = ['Cond', 'Time', 'Resp.O1', 'Resp.O2', 'Resp.O3']
    with open(path) as f:
        data = json.load(f)
    data = [[s[k] for k in cols] for u in data for s in u]
    data = pd.DataFrame(data=data, columns=cols)
    return data.rename(columns={'Cond': 'Effect'})


def split(X, y, test_size):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y.reshape(-1, 1), test_size=test_size)
    return [X_train, y_train, X_val, y_val]


features = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
epochs = 200
models = {
    # 'logistic_regression': lambda: LogisticRegression(multi_class='multinomial', solver='newton-cg'),
    # 'random_forest': lambda: RandomForestClassifier(n_estimators=100),
    # 'mlp': lambda: MLP(6, 3, 50, 100, 32),
    # 'mlp_sk': lambda: MLPClassifier(),
    'mlba_nn_0.001_10_wc_0_d_0_alpha_1_crim': {
        'data': 'Criminals',
        'model': lambda: MLBA_NN(6, 3, 10, epochs, 128, 0.001, weight_decay=0, dropout=0, alpha=1),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.2) + [True],
    },
    'mlba_nn_0.001_50_wc_0_d_0_alpha_1_crim': {
        'data': 'Criminals',
        'model': lambda: MLBA_NN(6, 3, 50, epochs, 128, 0.001, weight_decay=0, dropout=0, alpha=1),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.2) + [True],
    },
    'mlba_nn_0.001_10_wc_0_d_0_alpha_0.5_crim': {
        'data': 'Criminals',
        'model': lambda: MLBA_NN(6, 3, 10, epochs, 128, 0.001, weight_decay=0, dropout=0, alpha=0.5),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.2) + [True],
    },
    'mlba_nn_0.001_50_wc_0_d_0_alpha_0.5_crim': {
        'data': 'Criminals',
        'model': lambda: MLBA_NN(6, 3, 50, epochs, 128, 0.001, weight_decay=0, dropout=0, alpha=0.5),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.2) + [True],
    },
    'mlba_nn_0.001_10_wc_0_d_0_alpha_0.01_rect': {
        'data': 'Rectangles',
        'model': lambda: MLBA_NN(6, 3, 10, epochs, 128, 0.001, weight_decay=0, dropout=0, alpha=0.01),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.5) + [True],
    },
    'mlba_nn_0.001_10_wc_0_d_0_alpha_0.5_rect': {
        'data': 'Rectangles',
        'model': lambda: MLBA_NN(6, 3, 10, epochs, 128, 0.001, weight_decay=0, dropout=0, alpha=0.5),
        'params': lambda X, y: split(X, y.reshape(-1, 1), 0.5) + [True],
    },
}

experimentData = {
    'Rectangles': {
        'name': 'Rectangles',
        'train': pd.read_csv('data/E2.csv'),
        'test': {'e1a': pd.read_csv('data/E1a.csv'),
                 'e1b': pd.read_csv('data/E1b.csv'),
                 'e1c': pd.read_csv('data/E1c.csv')},
        # 'paper': {'e1a': load_paper_data('hb_mlba/e1a.pred.json'),
        #           'e1b': load_paper_data('hb_mlba/e1b.pred.json'),
        #           'e1c': load_paper_data('hb_mlba/e1c.pred.json'), }
    },
    'Criminals': {
        'name': 'Criminals',
        'train': pd.read_csv('data/E4.csv'),
        'test': {'e3a': pd.read_csv('data/E3a.csv'),
                 'e3b': pd.read_csv('data/E3b.csv'),
                 'e3c': pd.read_csv('data/E3c.csv')},
        # 'paper': {'e3a': load_paper_data('hb_mlba/e3a.pred.json'),
        #           'e3b': load_paper_data('hb_mlba/e3b.pred.json'),
        #           'e3c': load_paper_data('hb_mlba/e3c.pred.json'), }
    }
}


def get_freq(x):
    hist, _ = np.histogram(x, 3)
    return hist/hist.sum()


def predict(model, data):
    X_test = data[features].values
    y_test = data.response.values - 1
    actual_freq = get_freq(y_test)
    pred_freq = model.predict_proba(X_test).mean(0)
    return actual_freq, pred_freq, ((actual_freq - pred_freq) ** 2).mean()


def get_predictions(model, data):
    effects = data.groupby('Effect')
    effects_data = [d for _, d in effects]
    MSE = []
    actual = []
    pred = []
    for d in effects_data:
        actual_freq, pred_freq, mse = predict(model, d)
        MSE.append(mse)
        actual.append(actual_freq)
        pred.append(pred_freq)

    actual = np.array(actual)
    pred = np.array(pred)
    MSE = np.array(MSE)
    return actual, pred, MSE, [g for g, _ in effects]


markers = ['o', '^', 'd', 's', '.', '*', 'x', 'p', 'h', 'v']
colors = ['r', 'lime', 'b']


def save_results(path, actual, pred, mse, names, c=1):
    plt.figure(figsize=[4.8, 4.8])
    for i in range(3):
        x, y = actual[:, i], pred[:, i]
        for j in range(len(x)):
            plt.scatter(x[j], y[j], marker=markers[i], c=colors[c])
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.plot([0, 1], [0, 1])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(path.with_name(path.name + '.png'))
    plt.close()

    with path.with_name(path.name + '_mse.txt').open(mode='w') as f:
        for i in range(len(names)):
            f.write(f'{names[i]}: {mse[i]}\n')
        f.write(f'Overall: {mse.mean()}')


def run_model(m, exp, run):
    print(f'[{os.getpid()}] Run #{run} ...')
    np.random.seed(os.getpid() * 100 + run)
    X_train = exp['train'][features].values
    y_train = (exp['train'].response.values - 1)

    model = models[m]['model']()
    params = models[m]['params'](X_train, y_train)
    model.fit(*params)
    actual = {}
    pred = {}
    mse = {}
    names = {}
    for e in exp['test']:
        data = exp['test'][e]
        a, p, m, effects = get_predictions(model, data)
        actual[e] = a
        mse[e] = m
        pred[e] = p
        names[e] = effects

    return mse, actual, pred, names


def evaluate(m, n=10, jobs=5):
    exp = experimentData[models[m]['data']]
    print(f'[{os.getpid()}] Evaluating {m} on {exp["name"]} data...')
    dir = Path(f'out/res/{m}/{exp["name"]}')
    dir.mkdir(parents=True, exist_ok=True)

    args = [(m, exp, i+1) for i in range(n)]
    with Pool(jobs) as p:
        results = p.starmap(run_model, args)

    actual = {}
    pred = {}
    mse = {}
    names = {}

    for e in exp['test']:
        mse[e] = sum([r[0][e] for r in results]) / n
        actual[e] = sum([r[1][e] for r in results]) / n
        pred[e] = sum([r[2][e] for r in results]) / n
        names[e] = results[0][3][e]
        save_results(dir / f'{m}_{e}',
                     actual[e], pred[e], mse[e], names[e])


def run():
    for m in models:
        evaluate(m, n=50, jobs=50)


if __name__ == "__main__":
    run()
