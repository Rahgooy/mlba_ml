import torch
from torch.distributions import normal
from scipy.stats import norm
import numpy as np


class MLBA:
    def __init__(self, A, b, I0, beta, gamma, lambda1, lambda2, m):
        self.A = A
        self.b = b
        self.I0 = I0
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.m = m

    def get_drifts(self, x):
        return get_drifts(x, self.lambda1, self.lambda2, self.beta, self.gamma, self.I0, self.m)

    def sample(self, x, n):
        d = self.get_drifts(x)
        return sample_lba(n, self.b, self.A, d, torch.ones(d.shape))


def sample_lba(n, b, A, d, s, t0, st0=0, truncdrifts=True):
    pnorms = norm.pdf(-d)
    n_with_extras = int(np.ceil(n * (1 + 3 * np.prod(pnorms))))
    dist = torch.distributions.Normal(d, s)
    drifts = dist.sample((n_with_extras,)).view(-1, d.shape[1])
    drifts = drifts[(drifts > 0).all(1)].view(-1, d.shape[1])
    if truncdrifts:
        while drifts.shape[0] < n:
            temp = dist.sample((n_with_extras,)).view(-1, d.shape[1])
            drifts = torch.cat([drifts, temp])
            drifts = drifts[(drifts > 0).all(1)].view(-1, d.shape[1])
    drifts = drifts[:n]
    starts = torch.distributions.Uniform(0, A).sample((n, d.shape[1]))
    ttf = (b - starts) / drifts

    rt, resp = ttf.min(1)
    rt += t0 + \
        torch.distributions.Uniform(low=-st0/2, high=+st0/2).sample((n,))

    return rt, resp


def get_drifts(x, I0, m, lambda1, lambda2, gamma, beta):
    u = to_subjective(x, m)
    d1 = get_drift(u[:, 0], u[:, 1], u[:, 2],
                   I0, lambda1, lambda2, gamma, beta)
    d2 = get_drift(u[:, 1], u[:, 0], u[:, 2],
                   I0, lambda1, lambda2, gamma, beta)
    d3 = get_drift(u[:, 2], u[:, 0], u[:, 1],
                   I0, lambda1, lambda2, gamma, beta)
    d = torch.stack([d1, d2, d3], 1).view(x.shape[0], 3)
    return d.clamp(0, 1e6)


def get_drift(u, other1, other2, I0, lambda1, lambda2, gamma, beta):
    return gamma * val(u, other1, lambda1, lambda2, beta) + \
        gamma * val(u, other2, lambda1, lambda2, beta) + I0


def val(o1, o2, lambda1, lambda2, beta):
    return w1(o1[:, 0], o2[:, 0], lambda1, lambda2) * (o1[:, 0] - o2[:, 0]) + \
        w2(o1[:, 1], o2[:, 1], lambda1, lambda2, beta) * (o1[:, 1] - o2[:, 1])


def w1(o1, o2, lambda1, lambda2):
    return w2(o1, o2, lambda1, lambda2, 1.0)


def w2(o1, o2, lambda1, lambda2, beta):
    idx = o1 >= o2
    w = -beta * torch.abs(o1 - o2)
    w[idx] = torch.exp(lambda1 * w[idx])
    w[~idx] = torch.exp(lambda2 * w[~idx])
    return w


def to_subjective(x, m):
    x = x.reshape(x.shape[0], 3, 2)
    a = b = x.sum(2)
    # angle = torch.atan(x[:, :, 1] / x[:, :, 0])
    u1 = b / ((x[:, :, 1] / x[:, :, 0])**m + (b/a)**m)**(1/m)
    u2 = b * (1 - (u1/a)**m)**(1/m)
    return torch.stack([u1, u2], 2).view(x.shape[0], 3, 2)
