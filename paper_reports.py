import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import itertools
import pickle
from helpers import mse

rect_exp = ['e1a', 'e1b', 'e1c']
crim_exp = ['e3a', 'e3b', 'e3c']


def get_median_idx(a):
    return np.argsort(a)[0]


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
        pred = np.array(pred_list).mean(0)
        eff_results = {}
        for i, effect in enumerate(names):
            eff_results[effect] = {
                'pred': pred[i],
                'actual': actual[i],
                'paper_pred': paper_pred[i],
                'pred_mse': mse(actual[i], pred[i]),
                'paper_mse': mse(actual[i], paper_pred[i]),
                'count': counts[i]
            }
        res[e] = eff_results

    return res


crim_models = ['mlp_crim', 'mlba_nn_crim', 'mlba_nn_m_crim']
rec_models = ['mlp_rect', 'mlba_nn_rect', 'mlba_nn_m_rect']

markers = ['o', '^', 'd', 's', '.', 'p', 'h', 'v', '*', 'x']
colors = ['r', 'lime', 'b', 'orange']


def draw_results(crim, rect):
    fig, ax = plt.subplots(
        2, 3, sharey=True, sharex=True, figsize=(3 * 3, 2 * 3))

    def plot(ax, exp, model, print_paper, color, label=None, title=None, mse=None, size=40):
        m = model[exp]
        a = np.array([m[e]['actual'] for e in m])
        p1 = np.array([m[e]['pred'] for e in m])
        p2 = np.array([m[e]['paper_pred'] for e in m])
        for j in range(len(a)):
            for i in range(3):
                if i == 0 and j == 0:
                    ax.scatter(a[j, i], p1[j, i], marker=markers[j],
                               c=colors[color], label=label, s=size)
                    if print_paper:
                        ax.scatter(a[j, i], p2[j, i], marker=markers[j],
                                   c=colors[1], label='MLBA-HB', s=size)
                else:
                    ax.scatter(a[j, i], p1[j, i], marker=markers[j],
                               c=colors[color], s=size)
                    if print_paper:
                        ax.scatter(a[j, i], p2[j, i], marker=markers[j],
                                   c=colors[1], s=size)

        if print_paper:
            ax.set(xlabel='Actual', ylabel='Prediction')
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            ax.plot([0, 1], [0, 1], linewidth=1, c='grey',
                    transform=ax.transAxes, linestyle='--')
            ax.set_xlim([0, 0.8])
            ax.set_ylim([0, 0.8])
            f = {
                'size': 16,
                'weight': 600,
                'family': 'Arial'
            }
            ax.set_title(title, fontdict=f)
        ax.label_outer()

    def draw_mse(ax, exp, model1, model2, model3):
        m1 = model1[exp]
        m2 = model2[exp]
        m3 = model3[exp]
        mse1 = sum([m1[e]['pred_mse'] * m1[e]['count']
                    for e in m1]) / sum([m1[e]['count'] for e in m1])
        mse2 = sum([m2[e]['pred_mse'] * m2[e]['count']
                    for e in m2]) / sum([m2[e]['count'] for e in m2])
        mse3 = sum([m3[e]['pred_mse'] * m3[e]['count']
                    for e in m3]) / sum([m3[e]['count'] for e in m3])
        mse4 = sum([m1[e]['paper_mse'] * m1[e]['count']
                    for e in m1]) / sum([m1[e]['count'] for e in m1])

        ax.text(0.47, 0.18, f'MSE: {mse1*100:0.2f}%',
                color=colors[2], fontdict={'size': 11, 'weight': 'bold'})
        ax.text(0.47, 0.13, f'MSE: {mse4*100:0.2f}%',
                color=colors[1], fontdict={'size': 11, 'weight': 'bold'})
        ax.text(0.47, 0.08, f'MSE: {mse2*100:0.2f}%',
                color=colors[0], fontdict={'size': 11, 'weight': 'bold'})
        ax.text(0.47, 0.03, f'MSE: {mse3*100:0.2f}%',
                color=colors[3], fontdict={'size': 11, 'weight': 'bold'})

    s1 = 30
    s2 = 20
    plot(ax[0, 0], 'e1a', rect[0], True, 2, 'MLP', 'Attraction')
    plot(ax[0, 0], 'e1a', rect[1], False, 0, 'MLBA-NN', size=s1)
    plot(ax[0, 0], 'e1a', rect[2], False, 3, 'MLBA-NN-m', size=s2)
    draw_mse(ax[0, 0], 'e1a', rect[0], rect[1], rect[2])
    ax[0, 0].legend(loc="upper left", borderpad=0.25, handletextpad=0, borderaxespad=0,
                    bbox_to_anchor=[-0.04, 1], frameon=False, prop={'size': 11,
                                                                        'weight': 'bold', })

    plot(ax[0, 1], 'e1b', rect[0], True, 2, title='Compromise')
    plot(ax[0, 1], 'e1b', rect[1], False, 0, size=s1)
    plot(ax[0, 1], 'e1b', rect[2], False, 3, size=s2)
    draw_mse(ax[0, 1], 'e1b', rect[0], rect[1], rect[2])

    plot(ax[0, 2], 'e1c', rect[0], True, 2, title='Similarity')
    plot(ax[0, 2], 'e1c', rect[1], False, 0, size=s1)
    plot(ax[0, 2], 'e1c', rect[2], False, 3, size=s2)
    draw_mse(ax[0, 2], 'e1c', rect[0], rect[1], rect[2])

    plot(ax[1, 0], 'e3a', crim[0], True, 2)
    plot(ax[1, 0], 'e3a', crim[1], False, 0, size=s1)
    plot(ax[1, 0], 'e3a', crim[2], False, 3, size=s2)
    draw_mse(ax[1, 0], 'e3a', crim[0], crim[1], crim[2])

    plot(ax[1, 1], 'e3b', crim[0], True, 2)
    plot(ax[1, 1], 'e3b', crim[1], False, 0, size=s1)
    plot(ax[1, 1], 'e3b', crim[2], False, 3, size=s2)
    draw_mse(ax[1, 1], 'e3b', crim[0], crim[1], crim[2])

    plot(ax[1, 2], 'e3c', crim[0], True, 2)
    plot(ax[1, 2], 'e3c', crim[1], False, 0, size=s1)
    plot(ax[1, 2], 'e3c', crim[2], False, 3, size=s2)
    draw_mse(ax[1, 2], 'e3c', crim[0], crim[1], crim[2])

    f = {
        'size': 14,
        'weight': 'bold',
    }
    plt.text(0.8, 1.06, 'Perceptual', fontdict=f,
             rotation=-90, color='maroon')
    plt.text(0.8, 0.24, 'Inference', fontdict=f, rotation=-90, color='maroon')

    plt.subplots_adjust(wspace=0.11, hspace=0.05, top=0.955,
                        left=0.065, right=0.975, bottom=0.08)
    # plt.show()
    plt.savefig('out/res/reports.pdf')


def get_results():
    outDir = Path(f'out/res/')
    crim = [get_stats(outDir / 'Criminals' / crim_models[0], 'Criminals'),
            get_stats(outDir / 'Criminals' / crim_models[1], 'Criminals'),
            get_stats(outDir / 'Criminals' / crim_models[2], 'Criminals'), ]
    rec = [get_stats(outDir / 'Rectangles' / rec_models[0], 'Rectangles'),
           get_stats(outDir / 'Rectangles' / rec_models[1], 'Rectangles'),
           get_stats(outDir / 'Rectangles' / rec_models[2], 'Rectangles')]
    draw_results(crim, rec)


get_results()
