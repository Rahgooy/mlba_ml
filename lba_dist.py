from torch.distributions import Normal
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
from mlba import sample_lba

norm = Normal(0, 1)


def npdf(x):
    return 1/np.sqrt(2*np.pi) * torch.exp(-0.5 * x**2)


def ncdf(x):
    return norm.cdf(x)


def simps(f, a, b, n=50):
    if n % 2 == 1:
        n += 1
    dx = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    S = dx/3 * (y[0:-1:2].sum(0) + 4*y[1::2].sum(0) + y[2::2].sum(0))
    return S


class LBA:
    """
    This class provides functions related to the distribution of LBA

    Params:
    A: The end of starting points interval [0, A]
    b: The threshold
    d: The Array of Mean drift rates
    s: The Standard deviation of drift rates    """

    def __init__(self, A, b, d, s):
        self.A = A if A.dim() == 1 else A.view(-1)
        self.b = b if b.dim() == 1 else b.view(-1)
        self.s = s if s.dim() == 1 else s.view(-1)
        self.d = d if d.dim() == 2 else d.view(1, -1)
        self.nOpt = self.d.shape[1]
        self.nS = self.d.shape[0]

    def timeCDF(self, t, i):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.view(self.nS, -1)

        A, b, d, s = self.A, self.b, self.d[:, i].view(self.nS, 1), self.s
        A = A.view(self.nS, 1).repeat((1, t.shape[1]))
        b = b.view(self.nS, 1).repeat((1, t.shape[1]))
        s = s.view(self.nS, 1).repeat((1, t.shape[1]))

        p = 1
        p += (b - A - t*d) / A * ncdf((b - A - t*d) / (t*s))
        p -= (b - t*d) / A * ncdf((b - t*d) / (t*s))
        p += t*s / A * npdf((b - A - t*d) / (t*s))
        p -= t*s / A * npdf((b - t*d) / (t*s))

        return p

    def timePDF(self, t, i):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.view(self.nS, -1)

        A, b, d, s = self.A, self.b, self.d[:, i].view(self.nS, 1), self.s
        A = A.view(self.nS, 1).repeat((1, t.shape[1]))
        b = b.view(self.nS, 1).repeat((1, t.shape[1]))
        s = s.view(self.nS, 1).repeat((1, t.shape[1]))

        p = 0
        p -= d * ncdf((b - A - t * d)/(t*s))
        p += s * npdf((b - A - t * d)/(t*s))
        p += d * ncdf((b - t * d)/(t*s))
        p -= s * npdf((b - t * d)/(t*s))
        p /= A

        return p

    def firstTimePdf(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        res = [1] * self.nOpt
        cdf = []
        pdf = []
        for i in range(self.nOpt):
            cdf.append(self.timeCDF(t, i))
            pdf.append(self.timePDF(t, i))

        for i in range(self.nOpt):
            for j in range(self.nOpt):
                if i == j:
                    res[i] *= pdf[i]
                else:
                    res[i] *= 1 - cdf[j]
        return torch.stack(res, 1).view(-1, self.nOpt)

    def probs(self):
        res = simps(self.firstTimePdf, 1, 20, 40)
        if res.sum() > 0:
            res /= res.sum()
        return res


if __name__ == "__main__":
    A = torch.tensor([3.0, 4.0], requires_grad=True)
    b = torch.tensor([10.0, 10.0], requires_grad=True)
    d = torch.tensor([[2.1, 2.14, 2.18], [1.41, 1.22, 1.18]],
                     requires_grad=True)
    s = torch.tensor([1.0, 1.0], requires_grad=True)

    lba = LBA(A, b, d, s)
    upper = b.item() / d.max().item() * 4
    t = np.linspace(0.2, upper, 1000).reshape((-1, 1))

    p = lba.probs()
    p.sum().backward()
    print(A.grad)
    p = p.detach().numpy()
    print('Analytical: ', p/p.sum())

    rt, resp = sample_lba(100000, b.item(), A.item(),
                          d.detach().reshape(1, -1), torch.ones((1, 3)) * s.item(), 0)
    resp_freq, _ = np.histogram(resp, 3)
    print('Emprical: ', resp_freq/100000)

    f = lba.timePDF(torch.tensor(t.tolist()), 0)
    plt.plot(t, f.detach())
    plt.show()

    f = lba.timeCDF(torch.tensor(t.tolist()), 0)
    plt.plot(t, f.detach())
    plt.show()

    f = lba.firstTimePdf(torch.tensor(t.tolist()))
    plt.plot(t, f.detach())
    plt.show()
