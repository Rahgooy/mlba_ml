import unittest
from lba_dist import LBA
import numpy as np
import torch


class TestMLBA(unittest.TestCase):
    def test_timeCDF(self):
        A = torch.tensor([3.0, 4.0, 4.5])
        b = torch.tensor([10.0, 10.0, 11.0])
        d = torch.tensor(
            [[2.1, 2.14, 2.18], [1.41, 1.22, 1.18], [5.41, 1.22, 1.18]])
        s = torch.tensor([1.0, 1.0, 1.0])
        t = np.linspace(1, 20, d.shape[0] * 5).reshape(d.shape[0], -1)

        lba = LBA(A, b, d, s)
        cdf = lba.timeCDF(t, 1)

        for i in range(d.shape[0]):
            lba = LBA(A[i], b[i], d[i], s[i])
            cdf_i = lba.timeCDF(t[i].reshape(1, -1), 1)
            self.assertTrue(np.allclose(cdf[i], cdf_i), msg=f'cdf[{i}]')

    def test_timePDF(self):
        A = torch.tensor([3.0, 4.0, 4.5])
        b = torch.tensor([10.0, 10.0, 11.0])
        d = torch.tensor(
            [[2.1, 2.14, 2.18], [1.41, 1.22, 1.18], [5.41, 1.22, 1.18]])
        s = torch.tensor([1.0, 1.0, 1.0])
        t = np.linspace(1, 20, d.shape[0] * 5).reshape(d.shape[0], -1)

        lba = LBA(A, b, d, s)
        cdf = lba.timePDF(t, 1)

        for i in range(d.shape[0]):
            lba = LBA(A[i], b[i], d[i], s[i])
            cdf_i = lba.timePDF(t[i].reshape(1, -1), 1)
            self.assertTrue(np.allclose(cdf[i], cdf_i), msg=f'cdf[{i}]')


if __name__ == '__main__':
    unittest.main()
