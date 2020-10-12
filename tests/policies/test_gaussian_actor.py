import unittest

import numpy as np
import torch

from torchrl.policies import GaussianActor
from tests.policies.common import CommonModel


class TestGaussianActor(CommonModel):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.policy = GaussianActor(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            device='cpu',
            max_action=1.,
            units=[4, 4])

    def test_call(self):
        """Not fix sigma"""
        # Single input
        state = torch.rand(1,
                           self.continuous_env.observation_space.low.size,
                           dtype=torch.float32)
        self._test_call(state, (1, self.continuous_env.action_space.low.size),
                        torch.Size([1]))
        # Multiple inputs
        states = torch.rand(self.batch_size,
                            self.continuous_env.observation_space.low.size,
                            dtype=torch.float32)
        self._test_call(
            states,
            (self.batch_size, self.continuous_env.action_space.low.size),
            torch.Size([self.batch_size]))

    def test_compute_log_probs(self):
        """Not fix sigma"""
        # Single input
        state = torch.rand(1,
                           self.continuous_env.observation_space.low.size,
                           dtype=torch.float32)
        action = torch.rand(1,
                            self.continuous_env.action_space.low.size,
                            dtype=torch.float32)
        self._test_compute_log_probs(state, action, torch.Size([1]))
        # Multiple inputs
        states = torch.rand(self.batch_size,
                            self.continuous_env.observation_space.low.size,
                            dtype=torch.float32)
        actions = torch.rand(self.batch_size,
                             self.continuous_env.action_space.low.size,
                             dtype=torch.float32)
        self._test_compute_log_probs(states, actions,
                                     torch.Size([self.batch_size]))


if __name__ == '__main__':
    unittest.main()
