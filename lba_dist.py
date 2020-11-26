from torch.distributions import Normal
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
from mlba import sample_lba
from profiling import global_profiler as profiler, profile


def npdf(x):
    return 1/np.sqrt(2*np.pi) * torch.exp(-0.5 * x**2)


def ncdf(x):
    norm = Normal(0, 1)
    return norm.cdf(x)


def simps(f, a, b, n, m):
    if n % 2 == 1:
        n += 1
    dx = (b-a)/n
    x = np.linspace(a, b, n+1)
    x = x.reshape(-1, 1).repeat(m, 1).transpose()
    y = f(x)
    S = dx/3 * (y[0:-1:2].sum(0) + 4*y[1::2].sum(0) + y[2::2].sum(0))
    return S


@profile
def gauss(f, m, approx_n):
    """
    from : https://stackoverflow.com/a/37421753/1847988
    """
    x_, w_ = np.polynomial.legendre.leggauss(approx_n)

    def zero2MinusOneAndOne(x):
        """
        Changes the interval from [0, 1] to [-1, 1]
        """
        ones = x == 1
        if ones.sum() > 0:
            x[ones] -= 1e-12
            print('Warning: Encountered values equal to 1')

        y = f(x/(1-x))
        for i in range(y.shape[1]):
            y[:, i, :] /= ((1-x)**2).t()

        return y

    def inf2MinusOneAndOne(x):
        """
        Changes the interval from [0, inf) to [-1, 1]
        """
        return 0.5 * zero2MinusOneAndOne(x/2 + 1/2)

    x = x_.reshape(-1, 1).repeat(m, 1).transpose()
    w = w_.reshape(-1, 1).repeat(m, 1)
    x = torch.tensor(x)
    w = torch.tensor(w)
    y = inf2MinusOneAndOne(x)

    for i in range(y.shape[1]):
        y[:, i, :] *= w

    return y.sum(0)


class LBA:
    """
    This class provides functions related to the distribution of LBA

    Params:
    A: The end of starting points interval [0, A]
    b: The threshold
    d: The Array of Mean drift rates
    s: The Standard deviation of drift rates    """

    def __init__(self, A, b, d, s, aprroximation_degree=100):
        self.A = A if A.dim() == 1 else A.view(-1)
        self.b = b if b.dim() == 1 else b.view(-1)
        self.s = s if s.dim() == 1 else s.view(-1)
        self.d = d if d.dim() == 2 else d.view(1, -1)
        self.nOpt = self.d.shape[1]
        self.nS = self.d.shape[0]
        self.approx_n = aprroximation_degree

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

        return p.clamp(0)

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
        """[summary]

        Args:
            t ([matrix]): times

        Returns:
            [matrix]: nTimes * nOps * nS
        """
        res = [1] * self.nOpt
        cdf = []
        pdf = []
        for i in range(self.nOpt):
            cdf.append(self.timeCDF(t, i))
            pdf.append(self.timePDF(t, i))

        for i in range(self.nOpt):
            for j in range(self.nOpt):
                if i == j:
                    res[i] *= pdf[i].t()
                else:
                    res[i] *= 1 - cdf[j].t()
        return torch.stack(res, 1)

    @profile('LBA.probs')
    def probs(self):
        res = gauss(self.firstTimePdf, self.nS, self.approx_n).t()
        i = np.random.randint(0, self.nOpt)
        # Make sure sum == 1
        res[:, i] = 1 - (res[:, :i].sum(1) + res[:, i+1:].sum(1))
        return res


if __name__ == "__main__":
    A = torch.tensor([3.0, 4.0], requires_grad=True)
    b = torch.tensor([10.0, 10.0], requires_grad=True)
    d = torch.tensor([[1, 2.14, 2.9], [1.41, 1.22, 1.18]],
                     requires_grad=True)
    s = torch.tensor([1.0, 1.0], requires_grad=True)

    lba = LBA(A, b, d, s)
    upper = b.max().item() / d.max().item() * 4
    t = np.linspace(0.2, upper, 1000)
    t = t.reshape(-1, 1).repeat(A.shape[0], 1).transpose()

    p = lba.probs()
    (p.sum()).backward()
    print("grads: ")
    print(A.grad)
    print(b.grad)
    print(d.grad)
    print()
    p = p.detach().numpy()
    print('Analytical: ', p)

    rt, resp = sample_lba(100000, b[0].item(), A[0].item(),
                          d[0].detach().reshape(1, -1), torch.ones((1, 3)) * s[0].item(), 0)
    resp_freq, _ = np.histogram(resp, 3)
    print('Emprical: ', resp_freq/100000)

    for i in range(3):
        f = lba.timePDF(t, i)
        plt.plot(t[0], f[0].detach())
    plt.title('Probability density function of time to reach the threshold')
    plt.xlabel('Time')
    plt.ylabel('PDF')
    plt.show()

    for i in range(3):
        f = lba.timeCDF(t, i)
        plt.plot(t[0], f[0].detach())
    plt.title('Cumulative prob. function of time to reach the threshold')
    plt.xlabel('Time')
    plt.ylabel('CDF')
    plt.legend(['A', 'B', 'C'])
    plt.show()

    f = lba.firstTimePdf(t)
    plt.plot(t[0], f[:, :, 0].detach())
    plt.title('Probability density function of first to reach to threshold')
    plt.xlabel('Time')
    plt.ylabel('Prob. of choice')
    plt.legend(['A', 'B', 'C'])
    plt.show()
