from scipy import stats
import numpy as np


def jsd(dist1, dist2):
    """Jensen-Shanon Divergence"""
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    M = (dist1 + dist2) / 2
    jsd = (stats.entropy(dist1, M) + stats.entropy(dist2, M)) / 2

    return jsd


def kld(dist1, dist2):
    """KL Divergence"""
    return stats.entropy(dist1, dist2)


def mse(dist1, dist2):
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    return ((dist1 - dist2) ** 2).mean()

def rotate_options(X, y):
    X1 = X.copy()
    X1[:, 0] = X[:, 2]
    X1[:, 1] = X[:, 3]
    X1[:, 2] = X[:, 0]
    X1[:, 3] = X[:, 1]
    y1 = y.copy()
    y1[y == 0] = 1
    y1[y == 1] = 0
    return np.concatenate((X, X1)), np.concatenate((y, y1))


