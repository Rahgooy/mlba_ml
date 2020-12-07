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
