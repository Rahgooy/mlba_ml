import numpy as np
import pickle
from pathlib import Path
from scaler import CustomScaler, DummyScaler
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from mlp import MLP


def mlba_pred(data, x):
    C1, C2 = round(x[0, 4], 4), round(x[0, 5], 4)
    point = data[(data.C1 == C1) & (data.C2 == C2)]
    f = (point.p1 / (point.p2 + point.p1 + 1e-12)).values[0]
    res = [point.d1.values[0], point.d2.values[0], point.d3.values[0], f]
    res += [point.p1.values[0], point.p2.values[0], point.p3.values[0]]
    return np.array(res).reshape(-1)


def predict(m, scaler, x):
    x = scaler.transform(x)
    if isinstance(m, MLP):
        d = np.zeros(3).tolist()
        dist = m.predict_proba(x)[0]

    else:
        d = m.get_d(x).detach().numpy().reshape(-1).tolist()
        dist = m.predict_proba_mlba(x, 100)[0]
    p = dist[0] / (dist[0] + dist[1])
    return np.array(d + [p] + dist.tolist())


def mean_predict(models, x):
    pred = 0
    for model, scaler in models:
        pred += predict(model, scaler, x)
    return pred / len(models)


def get_mesh(A, B, start, end, step, pred):
    x = np.arange(start, end + step, step)
    x, y = np.meshgrid(x, x)
    z = np.zeros((x.shape[0], x.shape[1], 7))
    for i in range(x.shape[0]):
        print(i)
        for j in range(x.shape[1]):
            z[i, j] = pred(np.array(A + B + [x[i, j], y[i, j]]).reshape(-1, 6))
    return x, y, z


def draw(start, end, x, y, z, A, B, name, path='out/res/runs'):
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(13, 4))
    # Draw d values
    for i in range(3):
        CS = ax[i].contourf(x, y, z[:, :, i], cmap=cm.coolwarm)
        ax[i].plot([start, end], [end, start], linewidth=1, c='black')
        ax[i].scatter([A[0], B[0]], [A[1], B[1]], c='black')
        ax[i].text(A[0] + 0.1, A[1], 'A')
        ax[i].text(B[0] + 0.1, B[1], 'B')
        ax[i].set_xlabel('$C_1$')
        ax[i].set_ylabel('$C_2$')
    ax[0].set_title('$d_A$')
    ax[1].set_title('$d_B$')
    ax[2].set_title('$d_C$')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    fig.colorbar(CS, cax=cbar_ax)
    plt.savefig(f'{path}/{name}_d.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(13, 4))
    # Draw d values
    for i in range(3):
        CS = ax[i].contourf(x, y, z[:, :, i + 4],
                            cmap=cm.coolwarm, levels=np.arange(0, 1.01, 0.1))
        ax[i].plot([start, end], [end, start], linewidth=1, c='black')
        ax[i].scatter([A[0], B[0]], [A[1], B[1]], c='black')
        ax[i].text(A[0] + 0.1, A[1], 'A')
        ax[i].text(B[0] + 0.1, B[1], 'B')
        ax[i].set_xlabel('$C_1$')
        ax[i].set_ylabel('$C_2$')
    ax[0].set_title('$P_A$')
    ax[1].set_title('$P_B$')
    ax[2].set_title('$P_C$')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    fig.colorbar(CS, cax=cbar_ax)
    plt.savefig(f'{path}/{name}_dist.pdf')
    plt.close()

    #
    fig = plt.figure(figsize=(5, 4.5))
    CS = plt.contourf(x, y, z[:, :, 3], cmap=cm.coolwarm, levels=np.arange(0, 1.01, 0.1))  # pcolormesh
    plt.plot([start, end], [end, start], linewidth=1, c='black')
    plt.scatter([A[0], B[0]], [A[1], B[1]], c='black')
    plt.text(A[0] + 0.1, A[1], 'A')
    plt.text(B[0] + 0.1, B[1], 'B')
    plt.xlabel('$C_1$')
    plt.ylabel('$C_2$')
    plt.title('$p_A/(p_A + p_B)$')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(CS, cax=cbar_ax)
    plt.savefig(f'{path}/{name}_p.pdf')
    plt.close()


if __name__ == "__main__":
    step = 0.1
    ####################### Paper example #####################
    A = [4, 6]
    B = [6, 4]
    data = pd.read_csv('hb_mlba/mesh_paper_example.csv')
    x, y, z = get_mesh(
        A, B, 1, 9, step, lambda x: mlba_pred(data, x))
    draw(1, 9, x, y, z, A, B, 'paper_example', 'out/res')

    ######################### MLBA ###########################
    A = [3, 4]
    B = [4, 3]
    data = pd.read_csv('hb_mlba/mesh_E2.csv')
    x, y, z = get_mesh(
        A, B, 1, 6, step, lambda x: mlba_pred(data, x))
    draw(1, 6, x, y, z, A, B, 'mlba_E2', 'out/res')

    data = pd.read_csv('hb_mlba/mesh_E4.csv')
    x, y, z = get_mesh(
        A, B, 1, 6, step, lambda x: mlba_pred(data, x))
    draw(1, 6, x, y, z, A, B, 'mlba_E4', 'out/res')

    ######################## MLBA_NN #########################
    rect_models = []
    crim_models = []
    mlp_rect_models = []
    mlp_crim_models = []
    name = ''
    runs = 50
    for i in range(1, runs + 1):
        path = Path(f'out/temp/models/mlba_nn_rect{name}_run_{i}.pkl')
        with path.open('rb') as f:
            model, scaler = pickle.load(f)
            rect_models.append((model, scaler))
        x, y, z = get_mesh(
            A, B, 1, 6, step, lambda x: predict(model, scaler, x))
        draw(1, 6, x, y, z, A, B, f'mlba_nn{name}_E2_{i}')

        path = Path(f'out/temp/models/mlba_nn_crim{name}_run_{i}.pkl')
        with path.open('rb') as f:
            model, scaler = pickle.load(f)
            crim_models.append((model, scaler))
        x, y, z = get_mesh(
            A, B, 1, 6, step, lambda x: predict(model, scaler, x))
        draw(1, 6, x, y, z, A, B, f'mlba_nn{name}_E4_{i}')

        path = Path(f'out/temp/models/mlp_rect{name}_run_{i}.pkl')
        with path.open('rb') as f:
            model, scaler = pickle.load(f)
            mlp_rect_models.append((model, scaler))
        x, y, z = get_mesh(
            A, B, 1, 6, step, lambda x: predict(model, scaler, x))
        draw(1, 6, x, y, z, A, B, f'mlp{name}_E2_{i}')

        path = Path(f'out/temp/models/mlp_crim{name}_run_{i}.pkl')
        with path.open('rb') as f:
            model, scaler = pickle.load(f)
            mlp_crim_models.append((model, scaler))
        x, y, z = get_mesh(
            A, B, 1, 6, step, lambda x: predict(model, scaler, x))
        draw(1, 6, x, y, z, A, B, f'mlp{name}_E4_{i}')

    x, y, z = get_mesh(
        A, B, 1, 6, step, lambda x: mean_predict(rect_models, x))
    draw(1, 6, x, y, z, A, B, f'mlba_nn{name}_E2_mean', 'out/res')

    x, y, z = get_mesh(
        A, B, 1, 6, step, lambda x: mean_predict(crim_models, x))
    draw(1, 6, x, y, z, A, B, f'mlba_nn{name}_E4_mean', 'out/res')

    x, y, z = get_mesh(
        A, B, 1, 6, step, lambda x: mean_predict(mlp_rect_models, x))
    draw(1, 6, x, y, z, A, B, f'mlp_nn{name}_E2_mean', 'out/res')

    x, y, z = get_mesh(
        A, B, 1, 6, step, lambda x: mean_predict(mlp_crim_models, x))
    draw(1, 6, x, y, z, A, B, f'mlp_nn{name}_E4_mean', 'out/res')
