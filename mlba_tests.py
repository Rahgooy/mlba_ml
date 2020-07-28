import unittest
from mlba import to_subjective, w1, w2, val, get_drifts, sample_lba
import numpy as np
import torch


class TestMLBA(unittest.TestCase):
    def test_to_subjective(self):
        x = torch.tensor([
            [3, 1],
            [1.9, 2.1],
            [1.1, 2.9]
        ]).view(1, -1)
        expcted_u = np.array([
            [3.794733, 1.264911],
            [2.683653, 2.966143],
            [1.418617, 3.739990]
        ])
        u = to_subjective(x, 2).detach().numpy().reshape(3, 2)
        self.assertTrue(np.allclose(u, expcted_u))

        expcted_u = np.array([
            [3.996716, 1.332239],
            [3.291775, 3.638278],
            [1.514870, 3.993748]
        ])
        u = to_subjective(x, 5).detach().numpy().reshape(3, 2)
        self.assertTrue(np.allclose(u, expcted_u))

    def test_w1_w2(self):
        self.assertAlmostEqual(
            w1(torch.tensor(3.0), torch.tensor(5.0), .2, .3), 0.5488116)
        self.assertAlmostEqual(
            w2(torch.tensor(4.0), torch.tensor(2.0), .6, .3, .2), 0.7866279)

    def test_val(self):
        x = torch.tensor([
            [3, 1],
            [1.9, 2.1],
            [1.1, 2.9]
        ])
        v = val(x[0].view(1, -1), x[1].view(1, -1), .2, .4, .3).item()
        self.assertAlmostEqual(v,  -0.08120442, 5)

    def test_get_drifts(self):
        x = torch.tensor([
            [3, 1],
            [1.9, 2.1],
            [1.1, 2.9]
        ]).view(1, -1)
        d = get_drifts(x, 1, 2, .2, .4, .1, .3).detach().numpy().ravel()
        self.assertTrue(np.allclose(d, [0.9140869, 1.1100769, 1.1191032]))

    def test_sample_lba(self):
        N = 1000000
        x = torch.tensor([
            [3, 1],
            [1.9, 2.1],
            [1.1, 2.9]
        ]).view(1, -1)     
        d = get_drifts(x, 1, 2, .2, .4, .1, .3)
        rt, resp = sample_lba(N, 10, 3, d, torch.ones((1, 3)), 0)   
        mean_rt = rt.mean()
        resp_freq, _ = np.histogram(resp, 3)
        print(rt.min(), mean_rt, resp_freq/N)


if __name__ == '__main__':
    unittest.main()
