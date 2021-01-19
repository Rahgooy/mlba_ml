import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import itertools
import pickle
from helpers import mse

outDir = Path(f'out/temp/models')


def get_params(prefix):
    params = []
    for f in outDir.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            with f.open('rb') as file:
                m = pickle.load(file)[0]
                m_ = m.softplus(m.m).detach().numpy()[0]
                A = m.sigmoid(m.A).detach().numpy()[0] * 10
                b = m.sigmoid(m.b_).detach().numpy()[0] * 10 + A
                params.append([m_, A, b])
    return np.array(params)


p = get_params('mlba_nn_m_crim_run')

print(f'm for Crim(m, A, b): {p.mean(0)} ± {1.96 * p.std(0) / np.sqrt(p.shape[0])}')

p = get_params('mlba_nn_m_rect_run')
print(f'm for Rect(m, A, b): {p.mean(0)} ± {1.96 * p.std(0) / np.sqrt(p.shape[0])}')
