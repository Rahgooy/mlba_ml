import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from tabulate import tabulate
import itertools
import pickle
from helpers import jsd, mse

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
        pred = np.array(pred_list).mean(0)
        eff_results = {}
        mean_mse = np.array(mse_list).mean(0)
        se_mse = stats.sem(mse_list)
        jsd_list = [[jsd(a, p) for a, p in zip(actual, predicted)]
                    for predicted in pred_list]
        mean_jsd = np.array(jsd_list).mean(0)
        se_jsd = stats.sem(jsd_list)

        for i, effect in enumerate(names):
            eff_results[effect] = {
                'pred': pred[i],
                'actual': actual[i],
                'paper_pred': paper_pred[i],
                'pred_mse': mean_mse[i],
                'pred_se_mse': se_mse[i],
                'paper_mse': mse(actual[i], paper_pred[i]),
                'pred_jsd': mean_jsd[i],
                'pred_se_jsd': se_jsd[i],
                'paper_jsd': jsd(actual[i], paper_pred[i]),
                'count': counts[i]
            }
        res[e] = eff_results

    return res


crim_models = ['mlp_crim', 'mlba_nn_crim', 'mlba_nn_m_crim']
rec_models = ['mlp_rect', 'mlba_nn_rect', 'mlba_nn_m_rect']

markers = ['o', '^', 'd', 's', '.', 'p', 'h', 'v', '*', 'x']
colors = ['#007acc', '#7acc00', '#7a00cc', '#c33']


def plot_prob_prob(ax, exp, model, print_paper, color, label=None, title=None, mse=None, size=40):
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


def add_mse_text(ax, exp, model1, model2, model3):
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


def draw_metric(exps, model1, model2, model3, metric, ylabel, name):
    data = np.zeros((4, 3))
    err = np.zeros((3, 3))
    for i, exp in enumerate(exps):
        m1 = model1[exp]
        m2 = model2[exp]
        m3 = model3[exp]
        data[0, i] = sum([m1[e][f'pred_{metric}'] * m1[e]['count']
                          for e in m1]) / sum([m1[e]['count'] for e in m1])
        data[1, i] = sum([m2[e][f'pred_{metric}'] * m2[e]['count']
                          for e in m2]) / sum([m2[e]['count'] for e in m2])
        data[2, i] = sum([m3[e][f'pred_{metric}'] * m3[e]['count']
                          for e in m3]) / sum([m3[e]['count'] for e in m3])
        data[3, i] = sum([m1[e][f'paper_{metric}'] * m1[e]['count']
                          for e in m1]) / sum([m1[e]['count'] for e in m1])

        # 95% CI: 1.96 * SE(Standard Error)
        err[0, i] = sum([m1[e][f'pred_se_{metric}'] * 1.96 * m1[e]['count']
                         for e in m1]) / sum([m1[e]['count'] for e in m1])
        err[1, i] = sum([m2[e][f'pred_se_{metric}'] * 1.96 * m2[e]['count']
                         for e in m2]) / sum([m2[e]['count'] for e in m2])
        err[2, i] = sum([m3[e][f'pred_se_{metric}'] * 1.96 * m3[e]['count']
                         for e in m3]) / sum([m3[e]['count'] for e in m3])

    w = 0.7
    x = np.array([1, 6, 11]) * w
    _, ax = plt.subplots(figsize=(5, 3.5))

    plt.bar(x - 1.5*w, data[0], yerr=err[0], label='MLP',
            width=w, hatch='//', edgecolor='#111', color=colors[0], error_kw=dict(
                lw=1, capsize=3, ecolor='#111'))

    plt.bar(x + 1.5*w, data[3], label='MLBA-HB',
            width=w, hatch='.', edgecolor='#111', color=colors[1])

    plt.bar(x - 0.5*w, data[1], yerr=err[1], label='MLBA-NN',
            width=w, hatch='\\\\', edgecolor='#111', color=colors[2], error_kw=dict(
                lw=1, capsize=3, ecolor='#111'))

    plt.bar(x + 0.5*w, data[2], yerr=err[2], label='MLBA-NN-m',
            width=w, hatch='O', edgecolor='#111', color=colors[3], error_kw=dict(
                lw=1, capsize=3, ecolor='#111'))

    plt.ylabel(ylabel, fontweight='black', fontfamily='Arial')
    ax_style(ax)
    plt.xticks([-2] + x.tolist() + [11*w + 3], labels=['',
                                                       'Attaction', 'Compromise', 'Similarity', ''])
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./out/res/figures/{name}-{metric}.pdf')
    plt.close()


def ax_style(ax):
    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlim(0,8)

    yticks = ax.get_yticks().tolist()
    ax.set_ylim(yticks[0], ax.get_ylim()[1])

    xticks = ax.get_xticks().tolist()
    ax.set_xlim(xticks[0], xticks[-1])

    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))


def style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'


def draw_metric_plots(crim, rect):
    draw_metric(rect_exp, rect[0], rect[1],
                rect[2], 'mse', 'MSE', 'perceptual')
    draw_metric(rect_exp, rect[0], rect[1], rect[2],
                'jsd', 'JS-Divergence $D_{js}$', 'perceptual')

    draw_metric(crim_exp, crim[0], crim[1],
                crim[2], 'mse', 'MSE', 'inference')
    draw_metric(crim_exp, crim[0], crim[1], crim[2],
                'jsd', 'JS-Divergence $D_{js}$', 'inference')


def draw_prob_prob_all(crim, rect, add_mse=False):
    _, ax = plt.subplots(2, 3, sharey=True, sharex=True,
                         figsize=(3 * 3, 2 * 3))

    s1 = 40
    s2 = 40
    plot_prob_prob(ax[0, 0], 'e1a', rect[0], True, 2, 'MLP', 'Attraction')
    plot_prob_prob(ax[0, 0], 'e1a', rect[1], False, 0, 'MLBA-NN', size=s1)
    plot_prob_prob(ax[0, 0], 'e1a', rect[2], False, 3, 'MLBA-NN-m', size=s2)
    if add_mse:
        add_mse_text(ax[0, 0], 'e1a', rect[0], rect[1], rect[2])
    ax[0, 0].legend(loc="upper left", borderpad=0.25, handletextpad=0, borderaxespad=0,
                    bbox_to_anchor=[-0.04, 1], frameon=False, prop={'size': 11,
                                                                    'weight': 'bold', })

    plot_prob_prob(ax[0, 1], 'e1b', rect[0], True, 2, title='Compromise')
    plot_prob_prob(ax[0, 1], 'e1b', rect[1], False, 0, size=s1)
    plot_prob_prob(ax[0, 1], 'e1b', rect[2], False, 3, size=s2)
    if add_mse:
        add_mse_text(ax[0, 1], 'e1b', rect[0], rect[1], rect[2])

    plot_prob_prob(ax[0, 2], 'e1c', rect[0], True, 2, title='Similarity')
    plot_prob_prob(ax[0, 2], 'e1c', rect[1], False, 0, size=s1)
    plot_prob_prob(ax[0, 2], 'e1c', rect[2], False, 3, size=s2)
    if add_mse:
        add_mse_text(ax[0, 2], 'e1c', rect[0], rect[1], rect[2])

    plot_prob_prob(ax[1, 0], 'e3a', crim[0], True, 2)
    plot_prob_prob(ax[1, 0], 'e3a', crim[1], False, 0, size=s1)
    plot_prob_prob(ax[1, 0], 'e3a', crim[2], False, 3, size=s2)
    if add_mse:
        add_mse_text(ax[1, 0], 'e3a', crim[0], crim[1], crim[2])

    plot_prob_prob(ax[1, 1], 'e3b', crim[0], True, 2)
    plot_prob_prob(ax[1, 1], 'e3b', crim[1], False, 0, size=s1)
    plot_prob_prob(ax[1, 1], 'e3b', crim[2], False, 3, size=s2)
    if add_mse:
        add_mse_text(ax[1, 1], 'e3b', crim[0], crim[1], crim[2])

    plot_prob_prob(ax[1, 2], 'e3c', crim[0], True, 2)
    plot_prob_prob(ax[1, 2], 'e3c', crim[1], False, 0, size=s1)
    plot_prob_prob(ax[1, 2], 'e3c', crim[2], False, 3, size=s2)
    if add_mse:
        add_mse_text(ax[1, 2], 'e3c', crim[0], crim[1], crim[2])

    f = {
        'size': 14,
        'weight': 'bold',
    }
    plt.text(0.8, 1.06, 'Perceptual', fontdict=f,
             rotation=-90, color='maroon')
    plt.text(0.8, 0.24, 'Inference', fontdict=f, rotation=-90, color='maroon')

    plt.subplots_adjust(wspace=0.11, hspace=0.11, top=0.945,
                        left=0.085, right=0.965, bottom=0.085)
    # plt.show()
    if add_mse:
        plt.savefig('out/res/figures/prob_prob_mse.pdf')
    else:
        plt.savefig('out/res/figures/prob_prob.pdf')



def get_results():
    outDir = Path(f'out/res/')
    crim = [get_stats(outDir / 'Criminals' / crim_models[0], 'Criminals'),
            get_stats(outDir / 'Criminals' / crim_models[1], 'Criminals'),
            get_stats(outDir / 'Criminals' / crim_models[2], 'Criminals'), ]
    rec = [get_stats(outDir / 'Rectangles' / rec_models[0], 'Rectangles'),
           get_stats(outDir / 'Rectangles' / rec_models[1], 'Rectangles'),
           get_stats(outDir / 'Rectangles' / rec_models[2], 'Rectangles')]
    draw_prob_prob_all(crim, rec, True)
    draw_prob_prob_all(crim, rec, False)
    draw_metric_plots(crim, rec)


style()
get_results()
