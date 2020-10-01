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
    S = dx/3 * torch.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
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
        self.A = A
        self.b = b
        self.d = d
        self.s = s

    def timeCDF(self, t):
        """
        The CDF of time to reach the threshold b

        Args:
            t (float): time
        returns: Pr(T <= t) for each drift rate
        """

        A, b, d, s = self.A, self.b, self.d, self.s
        p = 1
        p += (b - A - t*d) / A * ncdf((b - A - t*d) / (t*s))
        p -= (b - t*d) / A * ncdf((b - t*d) / (t*s))
        p += t*s / A * npdf((b - A - t*d) / (t*s))
        p -= t*s / A * npdf((b - t*d) / (t*s))

        return p.clamp(0)

    def timePDF(self, t):
        A, b, d, s = self.A, self.b, self.d, self.s
        p = 0
        p -= d * ncdf((b - A - t * d)/(t*s))
        p += s * npdf((b - A - t * d)/(t*s))
        p += d * ncdf((b - t * d)/(t*s))
        p -= s * npdf((b - t * d)/(t*s))
        p /= A
        return p

    def firstTimePdf(self, t):
        cdf = self.timeCDF(t)
        pdf = self.timePDF(t)
        res = torch.ones(cdf.shape)
        for i in range(cdf.shape[1]):
            for j in range(cdf.shape[1]):
                if i == j:
                    res[:, i] *= pdf[:, i]
                else:
                    res[:, i] *= 1 - cdf[:, j]
        return res

    def probs(self):
        res = torch.zeros((self.d.shape[0],))
        for i in range(self.d.shape[0]):
            # np.ceil((self.b[0].item() - self.A[0].item()) / self.d[0, i].item())
            a = 0
            b = np.ceil(self.b[0].item() / self.d[i].item()) * 3
            print(b)
            res[i] = simps(lambda x: self.firstTimePdf(
                torch.tensor(x.reshape(-1, 1).tolist()))[:, i], a, b, b * 2)
        return res


if __name__ == "__main__":
    A = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([10.0], requires_grad=True)
    d = torch.tensor([2.1, 4.3, 1.8], requires_grad=True)
    s = torch.tensor([1.0], requires_grad=True)

    lba = LBA(A, b, d, s)
    upper = b.item() / d.max().item() * 4
    t = np.linspace(0, upper, 1000).reshape((-1, 1))

    p = lba.probs()
    print('Analytical: ', p.detach().numpy())

    rt, resp = sample_lba(100000, b.item(), A.item(),
                          d.detach().reshape(1, -1), torch.ones((1, 3)), 0)
    resp_freq, _ = np.histogram(resp, 3)
    print('Emprical: ', resp_freq/100000)

    f = lba.timePDF(torch.tensor(t.tolist()))
    plt.plot(t, f.detach())
    plt.show()

    f = lba.timeCDF(torch.tensor(t.tolist()))
    plt.plot(t, f.detach())
    plt.show()

    f = lba.firstTimePdf(torch.tensor(t.tolist()))
    plt.plot(t, f.detach())
    plt.show()
