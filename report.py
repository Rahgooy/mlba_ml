import pandas as pd
from pathlib import Path
from tabulate import tabulate

e1a = pd.read_csv('data/E1a.csv')
e1b = pd.read_csv('data/E1b.csv')
e1c = pd.read_csv('data/E1c.csv')
e3a = pd.read_csv('data/E3a.csv')
e3b = pd.read_csv('data/E3b.csv')
e3c = pd.read_csv('data/E3c.csv')

rect_exp = [('e1a', e1a), ('e1b', e1b), ('e1c', e1c)]
crim_exp = [('e3a', e3a), ('e3b', e3b), ('e3c', e3c)]


def get_counts(data):
    return {e[0]: len(e[1]) for e in data.groupby('Effect')}


def get_stats(model_path, exp):
    modelMSE = []
    exp_list = rect_exp if exp == 'Rectangles' else crim_exp
    for e, data in exp_list:
        f = f'{model_path.name}_{e}_mse.txt'
        f = model_path / f
        if not f.exists():
            continue
        counts = get_counts(data)
        total = sum(counts[k] for k in counts)
        effects = f.read_text().split('\n')
        mse = 0
        for effect in effects[:-1]:
            key, val = effect.split(':')
            m = float(val)
            mse += m * counts[key]
        modelMSE.append((mse/total, total))
    overall = sum([x[0] * x[1] for x in modelMSE]) / sum([x[1]
                                                          for x in modelMSE]) if len(modelMSE) else 0.0
    return modelMSE, overall


def print_results(exp):
    outDir = Path(f'out/res/{exp}')
    results = []
    exp_counts = None
    for model in outDir.iterdir():
        if model.is_dir():
            modelMSE, overall = get_stats(model, exp)
            if overall:
                res = [model.name] + [m[0] for m in modelMSE] + [overall]
                results.append(res)
            if exp_counts is None:
                exp_counts = [m[1] for m in modelMSE]

    print("\n")
    print(exp)
    print("Counts:", exp_counts)
    results = sorted(results, key=lambda x: x[-1])
    h = ['Model', "E1a", "E1b", "E1c", "Overall"] if exp == 'Rectangles' else [
        'Model', "E3a", "E3b", "E3c", "Overall"]
    print(tabulate(results, headers=h, tablefmt='fancy_grid', floatfmt=".4f"))


print_results('Criminals')
print_results('Rectangles')
