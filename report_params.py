import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import itertools
import pickle
from helpers import mse

outDir = Path(f'out/temp/models')

A1 = 3
A2 = 5

X = [A1, A2]
Z = [A2, A1]
Rx = [A1 - 0.3, A2 - 0.3]
Rz = [A2 - 0.3, A1 - 0.3]

x1 = np.array(X + Z + Rx).reshape(1, -1)
x2 = np.array(X + Z + Rz).reshape(1, -1)

def get_params(prefix):
    params = []
    ds = []
    ps = []
    for f in outDir.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            with f.open('rb') as file:
                m = pickle.load(file)[0]
                
                d1 = m.get_d(x1).detach().numpy().reshape(-1).tolist()
                d2 = m.get_d(x2).detach().numpy().reshape(-1).tolist()
                ds.append(d1 + d2)

                p1 = m.predict_proba(x1).reshape(-1).tolist()
                p2 = m.predict_proba(x2).reshape(-1).tolist()
                ps.append(p1 + p2)

                m_ = m.softplus(m.m).detach().numpy()[0]
                A = m.sigmoid(m.A).detach().numpy()[0] * 10
                b = m.sigmoid(m.b_).detach().numpy()[0] * 10 + A
                params.append([m_, A, b])
    return np.array(params), np.array(ds), np.array(ps)


p, ds, ps = get_params('mlba_nn_m_crim_run')

print('-' * 100)
print(f'm for Crim(m, A, b): {p.mean(0)} ± {1.96 * p.std(0) / np.sqrt(p.shape[0])}')
print(f'd1 for Crim: {ds[:, :3].mean(0)} -> {ps[:, :3].mean(0)}')
print(f'd2 for Crim: {ds[:, 3:6].mean(0)} -> {ps[:, 3:6].mean(0)}')
print('-' * 100)

p, ds, ps = get_params('mlba_nn_m_rect_run')
print(f'm for Rect(m, A, b): {p.mean(0)} ± {1.96 * p.std(0) / np.sqrt(p.shape[0])}')
print(f'd1 for Rect: {ds[:, :3].mean(0)} -> {ps[:, :3].mean(0)}')
print(f'd2 for Rect: {ds[:, 3:6].mean(0)} -> {ps[:, 3:6].mean(0)}')
