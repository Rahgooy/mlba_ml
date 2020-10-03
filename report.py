import pandas as pd
from pathlib import Path
from tabulate import tabulate

e1a = pd.read_csv('data/E1a.csv')
e1b = pd.read_csv('data/E1b.csv')
e1c = pd.read_csv('data/E1c.csv')

experiments = [('e1a', e1a), ('e1b', e1b), ('e1c', e1c)]


def get_counts(data):
    return {e[0]: len(e[1]) for e in data.groupby('Effect')}


outDir = Path('out')

results = []
exp_counts = None
for model in outDir.iterdir():
    if model.is_dir():
        modelMSE = []
        for e, data in experiments:
            f = f'{model.name}_{e}_mse.txt'
            f = model / f
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
        results.append([model.name] + [m[0] for m in modelMSE] + [overall])
        if exp_counts is None:
            exp_counts = [m[1] for m in modelMSE]

print("\n" * 5)
print("Counts:", exp_counts)
print(tabulate(results, headers=['Model', "E1a", "E1b", "E1c", "Overall"], tablefmt='fancy_grid', floatfmt=".4f"))
