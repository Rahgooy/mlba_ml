import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import itertools

rect_exp = ['e1a', 'e1b', 'e1c']
crim_exp = ['e3a', 'e3b', 'e3c']


def get_stats(model_path, exp):
    exp_list = rect_exp if exp == 'Rectangles' else crim_exp
    res = {}
    for e in exp_list:
        f = f'{model_path.name}_{e}_preds.txt'
        f = model_path / f
        if not f.exists():
            continue
        effects = f.read_text().split('\n')
        eff_results = {}
        for effect in effects:
            if effect.strip() == '':
                continue
            vals = effect.split(',')
            key = vals[0]
            actual = np.fromstring(vals[1].strip()[1:-1], dtype=float, sep=' ')
            pred = np.fromstring(vals[2].strip()[1:-1], dtype=float, sep=' ')
            paper_pred = np.fromstring(
                vals[3].strip()[1:-1], dtype=float, sep=' ')
            eff_results[key] = {
                'pred': pred,
                'actual': actual,
                'paper_pred': paper_pred
            }
        res[e] = eff_results

    return res


crim_models = ['mlp_64_crim_', 'mlba_nn_crim']
rec_models = ['mlp_32_rect', 'mlba_nn_rect_1024']

markers = ['o', '^', 'd', 's', '.', '*', 'x', 'p', 'h', 'v']
colors = ['r', 'lime', 'b']


def draw_results(crim, rect):
    fig, ax = plt.subplots(2, 3)

    def plot(ax, exp, model, print_paper, color, label):
        m = model[exp]
        a = np.array([m[e]['actual'] for e in m])
        p1 = np.array([m[e]['pred'] for e in m])
        p2 = np.array([m[e]['paper_pred'] for e in m])
        for i in range(3):
            if i == 0:
                ax.scatter(a[:, i], p1[:, i], marker=markers[0],
                           c=colors[color], label=label)
                if print_paper:
                    ax.scatter(a[:, i], p2[:, i], marker=markers[0],
                               c=colors[1], label='Paper')
            else:
                ax.scatter(a[:, i], p1[:, i], marker=markers[0],
                        c=colors[color])
                if print_paper:
                    ax.scatter(a[:, i], p2[:, i], marker=markers[0],
                            c=colors[1])

        if print_paper:
            ax.set(xlabel='Actual', ylabel='Prediction')
            ax.plot([0, 1], [0, 1])
            # ax.xlim(0, 1)
            # ax.ylim(0, 1)
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


def get_results(exp):
    outDir = Path(f'out/res/')
    crim = [get_stats(outDir / 'Criminals' / crim_models[0], 'Criminals'),
            get_stats(outDir / 'Criminals' / crim_models[1], 'Criminals')]
    rec = [get_stats(outDir / 'Rectangles' / rec_models[0], 'Rectangles'),
           get_stats(outDir / 'Rectangles' / rec_models[1], 'Rectangles')]
    draw_results(crim, rec)


get_results('Criminals')
