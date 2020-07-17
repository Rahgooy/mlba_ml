import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.distributions.constraints as constraints


class BayesianMLBA:
    def __init__(self, epochs):
        self.epochs = epochs
        self.priors = {}
        self.posteriors = {}

    def _sample_priors(self):
        self.priors['mu_A'] = pyro.sample(
            'mu_A', dist.Normal(2, 2), constraints.positive)
        self.priors['mu_b'] = pyro.sample(
            'mu_b', dist.Normal(2, 2), constraints.positive)

        self.priors['mu_I0'] = pyro.sample(
            'mu_I0', dist.Normal(1, 3), constraints.positive)
        self.priors['mu_beta'] = pyro.sample(
            'mu_beta', dist.Normal(2, 2), constraints.positive)

        self.priors['mu_beta'] = pyro.sample(
            'mu_lambda1', dist.Normal(0.3, 0.3), constraints.positive)
        self.priors['mu_beta'] = pyro.sample(
            'mu_lambda2', dist.Normal(0.3, 0.3), constraints.positive)

        self.priors['mu_t0'] = pyro.sample(
            'mu_t0', dist.Normal(0.5, 0.5), constraints.positive)

        self.priors['mu_m'] = pyro.sample(
            'mu_m', dist.Normal(1, 0.5), constraints.positive)

        self.priors['mu_gamma'] = pyro.sample(
            'mu_gamma', dist.Normal(2, 3), constraints.positive)

        vars = ['A', 'b', 't0', 'I0', 'lambda1', 'lambda2',
                'sigma_beta', 'sigma_m', 'sigma_lambda']
        for v in vars:
            self.priors[f'sigma_{v}'] = pyro.sample(
                f'sigma_{v}', dist.Gamma(0.001, 0.001))

    def _sample_group_level_posterior(self, x):
        priors = self.priors
        with pyro.plate('data'):
            self.posteriors['A_i'] = pyro.sample('A_i', dist.Normal(
                priors['mu_A'], priors['sigma_A']), constraints.greater_than_eq(0.0))

            self.posteriors['b_i - A_i'] = pyro.sample('bi - A_i', dist.Normal(
                priors['mu_b'], priors['sigma_b']), constraints.greater_than_eq(0.0))

            self.posteriors['t0_i'] = pyro.sample('t0_i', dist.Normal(
                priors['mu_t0'], priors['sigma_t0']), constraints.greater_than_eq(0.0))

            self.posteriors['I0_i'] = pyro.sample('I0_i', dist.Normal(
                priors['mu_I0'], priors['sigma_I0']), constraints.greater_than_eq(0.0))

            self.posteriors['lambda1_i'] = pyro.sample('lambda1_i', dist.Normal(
                priors['mu_lambda1'], priors['sigma_lambda1']), constraints.greater_than_eq(0.0))

            self.posteriors['lambda2_i'] = pyro.sample('lambda2_i', dist.Normal(
                priors['mu_lambda2'], priors['sigma_lambda2']), constraints.greater_than_eq(0.0))

            self.posteriors['beta_i'] = pyro.sample('beta_i', dist.Normal(
                priors['mu_beta'], priors['sigma_beta']), constraints.greater_than_eq(0.0))

            self.posteriors['m_i'] = pyro.sample('m_i', dist.Normal(
                priors['mu_m'], priors['sigma_m']), constraints.interval(0.2, 5))

            self.posteriors['gamma_i'] = pyro.sample('gamma_i', dist.Normal(
                priors['mu_gamma'], priors['sigma_gamma']), constraints.greater_than_eq(0.0))

    def _get_drifts(self, x, I_0, m, lambda1, lambda2, gamma, beta):
        u = self._to_subjective(x)
        d1 = self._get_drift(
            gamma, u[:, 0], u[:, 1], u[:, 2], lambda1, lambda2, beta, I0)
        d2 = self._get_drift(
            gamma, u[:, 1], u[:, 0], u[:, 2], lambda1, lambda2, beta, I0)
        d3 = self._get_drift(
            gamma, u[:, 2], u[:, 0], u[:, 1], lambda1, lambda2, beta, I0)

        d = torch.stack([d1, d2, d3], 1).view(x.shape[0], 3)
        return d.clamp(0, 1e6)

    def _get_drift(self, gamma, u, other1, other2, lambda1, lambda2, beta, I0):
        return gamma * self._val(u, other1, lambda1, lambda2, beta) + \
            gamma * self._val(u, other2, lambda1, lambda2, beta) + I_0

    def _val(self, o1, o2, lambda1, lambda2, beta):
        return
        self.w1(o1[:, 0], o2[:, 0], lambda1, lambda2) * (o1[:, 1] - o2[:, 1]) +
        self.w2(o1[:, 2], o2[:, 2], lambda1, lambda2, beta) * \
            (o1[:, 2] - o2[:, 2])

    def _to_subjective(self, x, m):
        x = x.reshape(x.shape[0], 3, 2)
        a = b = x.sum(2)
        angle = torch.atan(x[:, :, 0] / x[:, :, 1])
        u1 = b / (torch.tan(angle)**m + (b/a)**m)**(1/m)
        u2 = b * (1 - (u1/a)**m)**(1/m)
        return torch.stack([u1, u2], 1).view(x.shape[0], 3, 2)

    def _sample_MLBA(self, x, y):
        with pyro.plate('data'):

    def fit(X, Y):
        X = np.array(X)
        Y = np.array(Y)
