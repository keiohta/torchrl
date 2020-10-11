import unittest

import gym
import numpy as np
import torch


class CommonAlgos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.discrete_env = gym.make('CartPole-v0')
        cls.continuous_env = gym.make("Pendulum-v0")
        cls.batch_size = 32
        cls.agent = None
        cls.device = 'cpu'


class CommonOffPolAlgos(CommonAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = None
        cls.action_dim = None
        cls.is_discrete = True

    def test_get_action(self):
        if self.agent is None:
            return
        # Single input
        state = self.env.reset()
        action_train = self.agent.get_action(state, test=False)
        action_test = self.agent.get_action(state, test=True)
        if self.is_discrete:
            self.assertTrue(
                isinstance(action_train, (torch.int32, torch.int64, int)))
            self.assertTrue(
                isinstance(action_test, (torch.int32, torch.int64, int)))
        else:
            self.assertEqual(action_train.shape[0], self.action_dim)
            self.assertEqual(action_test.shape[0], self.action_dim)

        # Multiple inputs
        states = torch.zeros([self.batch_size, state.shape[0]],
                             dtype=torch.float32)
        actions_train = self.agent.get_action(states, test=False)
        actions_test = self.agent.get_action(states, test=True)

        if self.is_discrete:
            self.assertEqual(actions_train.shape, (self.batch_size, ))
            self.assertEqual(actions_test.shape, (self.batch_size, ))
        else:
            self.assertEqual(actions_train.shape,
                             (self.batch_size, self.action_dim))
            self.assertEqual(actions_test.shape,
                             (self.batch_size, self.action_dim))

    def test_get_action_greedy(self):
        if self.agent is None:
            return
        # Multiple inputs
        states = np.zeros(shape=(self.batch_size,
                                 self.env.reset().astype(np.float32).shape[0]),
                          dtype=np.float32)
        actions_train = self.agent.get_action(states, test=False)
        actions_test = self.agent.get_action(states, test=True)

        # All actions should be same if `test=True`, and not same if `test=False`
        if self.is_discrete:
            self.assertEqual(torch.prod(torch.unique(actions_test).shape), 1)
            self.assertGreater(torch.prod(torch.unique(actions_train).shape),
                               1)
        else:
            self.assertEqual(
                torch.prod(
                    torch.all(actions_test == actions_test[0, :], axis=0)), 1)
            self.assertEqual(
                torch.prod(
                    torch.all(actions_train == actions_train[0, :], axis=0)),
                0)

    def test_train(self):
        if self.agent is None:
            return
        rewards = torch.zeros([self.batch_size, 1], dtype=torch.float32)
        dones = torch.zeros([self.batch_size, 1], dtype=torch.float32)
        obses = torch.zeros(
            [self.batch_size, self.env.observation_space.shape[0]],
            dtype=torch.float32)
        acts = torch.zeros([self.batch_size, self.action_dim],
                           dtype=torch.float32)
        with torch.autograd.set_detect_anomaly(True):
            self.agent.train(obses, acts, obses, rewards, dones)

    def test_compute_td_error(self):
        if self.agent is None:
            return
        rewards = torch.from_numpy(
            np.zeros(shape=(self.batch_size, 1), dtype=np.float32))
        dones = torch.from_numpy(
            np.zeros(shape=(self.batch_size, 1), dtype=np.float32))
        obses = torch.from_numpy(
            np.zeros(shape=(self.batch_size, ) +
                     self.env.observation_space.shape,
                     dtype=np.float32))
        acts = torch.from_numpy(
            np.zeros(shape=(
                self.batch_size,
                self.continuous_env.action_space.low.size,
            ),
                     dtype=np.float32))
        self.agent.compute_td_error(states=obses,
                                    actions=acts,
                                    next_states=obses,
                                    rewards=rewards,
                                    dones=dones)


class CommonOffPolContinuousAlgos(CommonOffPolAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = cls.continuous_env
        cls.action_dim = cls.continuous_env.action_space.low.size
        cls.is_discrete = False


class CommonOffPolDiscreteAlgos(CommonOffPolAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = cls.discrete_env
        cls.action_dim = 1
        cls.is_discrete = True


if __name__ == '__main__':
    unittest.main()
