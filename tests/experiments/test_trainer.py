import unittest

import gym

from torchrl.algos import DQN
from torchrl.experiments import RLTrainer


class TestRLTrainer(unittest.TestCase):
    def test_empty_args(self):
        """
        Test empty args {}
        """
        env = gym.make("Pendulum-v0")
        test_env = gym.make("Pendulum-v0")
        policy = DQN(state_shape=env.observation_space.shape,
                     action_dim=env.action_space.high.size,
                     device='cpu',
                     memory_capacity=1000,
                     batch_size=32,
                     n_warmup=10)
        RLTrainer(policy, env, 'cpu', {}, test_env=test_env)

    def test_with_args(self):
        """
        Test with args
        """
        max_steps = 400
        env = gym.make("Pendulum-v0")
        test_env = gym.make("Pendulum-v0")
        policy = DQN(state_shape=env.observation_space.shape,
                     action_dim=env.action_space.high.size,
                     device='cpu',
                     memory_capacity=1000,
                     batch_size=32,
                     n_warmup=10)
        trainer = RLTrainer(policy,
                            env,
                            'cpu', {"max_steps": max_steps},
                            test_env=test_env)
        self.assertEqual(trainer._max_steps, max_steps)

    def test_invalid_args(self):
        """
        Test with invalid args
        """
        env = gym.make("Pendulum-v0")
        test_env = gym.make("Pendulum-v0")
        policy = DQN(state_shape=env.observation_space.shape,
                     action_dim=env.action_space.high.size,
                     device='cpu',
                     memory_capacity=1000,
                     batch_size=32,
                     n_warmup=10)
        with self.assertRaises(ValueError):
            RLTrainer(policy,
                      env,
                      'cpu', {"NOT_EXISTING_OPTIONS": 1},
                      test_env=test_env)


if __name__ == "__main__":
    unittest.main()
