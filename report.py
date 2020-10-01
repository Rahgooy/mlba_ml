import pandas as pd
from pathlib import Path

e1a = pd.read_csv('data/E1a.csv')
e1b = pd.read_csv('data/E1b.csv')
e1c = pd.read_csv('data/E1c.csv')

experiments = [('e1a', e1a), ('e1b', e1b), ('e1c', e1c)]


def get_counts(data):
    return {e[0]: len(e[1]) for e in data.groupby('Effect')}


outDir = Path('out')

for model in outDir.iterdir():
    if model.is_dir():
        modelMSE = []
        for e, data in experiments:
            counts = get_counts(data)
            total = sum(counts[k] for k in counts)
            f = f'{model.name}_{e}_mse.txt'
            f = model / f
            effects = f.read_text().split('\n')
            mse = 0
            for effect in effects[:-1]:
                key, val = effect.split(':')
                m = float(val)
                mse += m * counts[key]
            modelMSE.append((mse/total, total))
        r = model.name + ': ' + ' & '.join([f'{x[0]:0.4f}' for x in modelMSE])
        overall = sum([x[0] * x[1] for x in modelMSE]) / sum([x[1] for x in modelMSE])
        r += f'& {overall:0.4f}'
        print(r)
        print(f'Counts: {[x[1] for x in modelMSE]}')
        
