import unittest

import gym
import numpy as np
from parameterized import parameterized
import torch
import torch.nn as nn

from torchrl.policies import GaussianMLPPolicy
from tests.tools.dummy_env import DummyBoxEnv, DummyDictEnv, GymEnv


class TestGaussianMLPPolicy(unittest.TestCase):
    @parameterized.expand([[(1, )], [(2, )], [(3, )], [(1, 4)], [(3, 5)]])
    def test_get_action(self, units):
        env = GymEnv(DummyBoxEnv())
        states_dim = env.observation_space.flat_dim
        act_dim = env.action_space.flat_dim
        states = torch.ones(states_dim, dtype=torch.float)
        init_std = 2.

        policy = GaussianMLPPolicy(
            states_dim,
            act_dim,
            device='cpu',
            units=units,
            init_std=init_std,
            hidden_nonlinearity=None,
            w_init=nn.init.ones_,
        )

        dist, log = policy(states)

        expected_mean = torch.full(
            (act_dim, ),
            states_dim * (torch.Tensor(units).prod().item()),
            dtype=torch.float)
        expected_variance = init_std**2
        action, prob = policy.get_action(states)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((act_dim, ), expected_variance, dtype=torch.float))
        assert action.shape == (act_dim, )

    @parameterized.expand([[1, (1, )], [5, (3, )], [8, (4, )], [15, (1, 2)],
                           [30, (3, 4, 10)]])
    def test_get_action_batch(self, batch_size, units):
        env = GymEnv(DummyBoxEnv())
        states_dim = env.observation_space.flat_dim
        act_dim = env.action_space.flat_dim
        states = torch.ones([batch_size, states_dim], dtype=torch.float32)
        init_std = 2.

        policy = GaussianMLPPolicy(states_dim,
                                   act_dim,
                                   device='cpu',
                                   units=units,
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   w_init=nn.init.ones_)

        dist = policy(states)[0]

        expected_mean = torch.full([batch_size, act_dim],
                                   states_dim *
                                   (torch.Tensor(units).prod().item()),
                                   dtype=torch.float)
        expected_variance = init_std**2
        action, prob = policy.get_actions(states)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((batch_size, act_dim),
                       expected_variance,
                       dtype=torch.float))
        assert action.shape == (batch_size, act_dim)

    def test_get_action_dict_space(self):
        env = GymEnv(DummyDictEnv(obs_space_type='box', act_space_type='box'))
        states_dim = env.observation_space.flat_dim
        act_dim = env.action_space.flat_dim
        policy = GaussianMLPPolicy(states_dim,
                                   act_dim,
                                   device='cpu',
                                   units=(1, ),
                                   hidden_nonlinearity=None,
                                   w_init=nn.init.ones_)
        states = env.reset()[0]
        states = env.observation_space.flatten(states)

        action, _ = policy.get_action(states)
        assert env.action_space.shape == action.shape

        actions, _ = policy.get_actions(np.array([states, states]))
        for action in actions:
            assert env.action_space.shape == action.shape
        actions, _ = policy.get_actions(np.array([states, states]))
        for action in actions:
            assert env.action_space.shape == action.shape


if __name__ == '__main__':
    unittest.main()
