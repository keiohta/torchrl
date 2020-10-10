import unittest

import numpy as np
import torch

from torchrl.distributions import DiagonalGaussian
from tests.distributions.common import CommonDist


class TestDiagonalGaussian(CommonDist):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dist = DiagonalGaussian(dim=cls.dim)
        cls.param = {
            "mean":
            torch.zeros([1, cls.dim], dtype=torch.float32),
            "log_std":
            torch.ones([1, cls.dim], dtype=torch.float32) *
            torch.log(torch.tensor(1.))
        }
        cls.params = {
            "mean":
            torch.zeros([cls.batch_size, cls.dim], dtype=torch.float32),
            "log_std":
            torch.ones([cls.batch_size, cls.dim], dtype=torch.float32) *
            torch.log(torch.tensor(1.))
        }

    def test_kl(self):
        # KL of same distribution should be zero
        test_kl_1 = self.dist.kl(self.param, self.param)
        test_kl_2 = self.dist.kl(self.params, self.params)
        np.testing.assert_array_equal(test_kl_1, np.zeros(shape=(1, )))
        np.testing.assert_array_equal(test_kl_2,
                                      np.zeros(shape=(self.batch_size, )))

    def test_log_likelihood(self):
        pass

    def test_ent(self):
        pass

    def test_sample(self):
        samples = self.dist.sample(self.param)
        self.assertEqual(samples.shape, (1, self.dim))
        samples = self.dist.sample(self.params)
        self.assertEqual(samples.shape, (self.batch_size, self.dim))


if __name__ == '__main__':
    unittest.main()
