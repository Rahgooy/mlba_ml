import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
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


features = ['Rect1Height', 'Rect1Width', 'Rect2Height',
            'Rect2Width', 'Rect3Height', 'Rect3Width']
epochs = 200
models = {
    # 'logistic_regression': lambda: LogisticRegression(multi_class='multinomial', solver='newton-cg'),
    # 'random_forest': lambda: RandomForestClassifier(n_estimators=100),
    # 'mlp': lambda: MLP(6, 3, 50, 100, 32),
    # 'mlp_sk': lambda: MLPClassifier(),
    'mlba_nn_0.01_32': lambda: MLBA_NN(6, 3, 50, epochs, 32, 0.01),
    'mlba_nn_0.005_32': lambda: MLBA_NN(6, 3, 50, epochs, 32, 0.005),
    'mlba_nn_0.0005_32': lambda: MLBA_NN(6, 3, 50, epochs, 32, 0.0005),
    'mlba_nn_0.001_32': lambda: MLBA_NN(6, 3, 50, epochs, 32, 0.001),
    'mlba_nn_0.001_64': lambda: MLBA_NN(6, 3, 50, epochs, 64, 0.001),
    'mlba_nn_0.001_128': lambda: MLBA_NN(6, 3, 50, epochs, 128, 0.001),
    'mlba_nn_0.001_32_100': lambda: MLBA_NN(6, 3, 100, epochs, 32, 0.001),
}

train_data = pd.read_csv('data/E2.csv')
X_train = train_data[features].values
y_train = (train_data.response.values - 1)

e1a = pd.read_csv('data/E1a.csv')
e1b = pd.read_csv('data/E1b.csv')
e1c = pd.read_csv('data/E1c.csv')

experimentData = {
    'e1a': (e1a, load_paper_data('hb_mlba/e1a.pred.json')),
    'e1b': (e1b, load_paper_data('hb_mlba/e1b.pred.json')),
    'e1c': (e1c, load_paper_data('hb_mlba/e1c.pred.json')),
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


def save_results(path, actual, pred, mse, names, c):
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


def evaluate(model_creator, name, n=10, c=1):
    print(f'[{os.getpid()}] Evaluating {name} ...')
    dir = Path(f'out/{name}')
    dir.mkdir(parents=True, exist_ok=True)
    actual = {}
    pred = {}
    mse = {}
    names = {}
    for i in range(n):
        print(f'[{os.getpid()}] Iteration {i} ...')
        model = model_creator()
        model.fit(X_train, y_train)
        for e in experimentData:
            data = experimentData[e][0]
            a, p, m, effects = get_predictions(model, data)

            if e not in actual:
                actual[e] = mse[e] = 0

            actual[e] += a
            mse[e] += m
            pred[e] = p
            names[e] = effects

    for e in experimentData:
        mse[e] /= n
        actual[e] /= n
        save_results(dir / f'{name}_{e}',
                     actual[e], pred[e], mse[e], names[e], c)


def eval_process(m):
    evaluate(models[m], m, n=50)


with Pool(5) as p:
    p.map(eval_process, list(models.keys()))
