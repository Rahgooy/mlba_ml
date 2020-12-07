import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import itertools
import pickle

rect_exp = ['e1a', 'e1b', 'e1c']
crim_exp = ['e3a', 'e3b', 'e3c']


def get_median_idx(a):
    return np.argsort(a)[len(a) // 2]


def get_stats(model_path, exp):
    exp_list = rect_exp if exp == 'Rectangles' else crim_exp
    res = {}
    for e in exp_list:
        f = f'{model_path.name}_{e}_preds.pkl'
        f = model_path / f
        if not f.exists():
            continue
        with f.open('rb') as file:
            (names, actual, pred_list, paper_pred,
             mse_list, counts) = pickle.load(file)
        overallMSE = [sum(m * np.array(counts)) / sum(counts)
                      for m in mse_list]
        md = get_median_idx(overallMSE)
        pred = pred_list[md]  # np.array(pred_list).mean(0)
        eff_results = {}
        for i, effect in enumerate(names):
            eff_results[effect] = {
                'pred': pred[i],
                'actual': actual[i],
                'paper_pred': paper_pred[i]
            }
        res[e] = eff_results

    return res


crim_models = ['mlp_crim', 'mlba_nn_crim']
rec_models = ['mlp_rect', 'mlba_nn_rect']

markers = ['o', '^', 'd', 's', '.', '*', 'x', 'p', 'h', 'v']
colors = ['r', 'lime', 'b']


def draw_results(crim, rect):
    fig, ax = plt.subplots(2, 3)

    def plot(ax, exp, model, print_paper, color, label):
        m = model[exp]
        a = np.array([m[e]['actual'] for e in m])
        p1 = np.array([m[e]['pred'] for e in m])
        p2 = np.array([m[e]['paper_pred'] for e in m])
        size = 10
        for i in range(3):
            if i == 0:
                ax.scatter(a[:, i], p1[:, i], marker=markers[i],
                           c=colors[color], label=label, s=size)
                if print_paper:
                    ax.scatter(a[:, i], p2[:, i], marker=markers[i],
                               c=colors[1], label='Paper', s=size)
            else:
                ax.scatter(a[:, i], p1[:, i], marker=markers[i],
                           c=colors[color], s=size)
                if print_paper:
                    ax.scatter(a[:, i], p2[:, i], marker=markers[i],
                               c=colors[1], s=size)

        if print_paper:
            ax.set(xlabel='Actual', ylabel='Prediction')
            ax.plot([0, 1], [0, 1], linewidth=1, c='black')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            # ax.title(title)

        ax.label_outer()

    plot(ax[0, 0], 'e1a', rect[0], True, 2, 'MLP')
    plot(ax[0, 0], 'e1a', rect[1], False, 0, 'MLBA_NN')
    ax[0, 0].legend(loc="upper left", fontsize='x-small')
    plot(ax[0, 1], 'e1b', rect[0], True, 2, None)
    plot(ax[0, 1], 'e1b', rect[1], False, 0, None)
    plot(ax[0, 2], 'e1c', rect[0], True, 2, None)
    plot(ax[0, 2], 'e1c', rect[1], False, 0, None)

    plot(ax[1, 0], 'e3a', crim[0], True, 2, None)
    plot(ax[1, 0], 'e3a', crim[1], False, 0, None)
    plot(ax[1, 1], 'e3b', crim[0], True, 2, None)
    plot(ax[1, 1], 'e3b', crim[1], False, 0, None)
    plot(ax[1, 2], 'e3c', crim[0], True, 2, None)
    plot(ax[1, 2], 'e3c', crim[1], False, 0, None)

    plt.show()
    print()


def get_results():
    outDir = Path(f'out/res/')
    crim = [get_stats(outDir / 'Criminals' / crim_models[0], 'Criminals'),
            get_stats(outDir / 'Criminals' / crim_models[1], 'Criminals')]
    rec = [get_stats(outDir / 'Rectangles' / rec_models[0], 'Rectangles'),
           get_stats(outDir / 'Rectangles' / rec_models[1], 'Rectangles')]
    draw_results(crim, rec)


get_results()
