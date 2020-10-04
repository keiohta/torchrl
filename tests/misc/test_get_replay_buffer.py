import unittest

import gym

from cpprb import ReplayBuffer
from cpprb import PrioritizedReplayBuffer

from torchrl.misc import get_replay_buffer
from torchrl.policies import OffPolicyAgent


class TestGetReplayBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 32
        cls.memory_capacity = 32
        # TODO(sff1019): Add OnPolicyAgent
        cls.off_policy_agent = OffPolicyAgent(
            name="OffPolicyAgent", memory_capacity=cls.memory_capacity)
        cls.discrete_env = gym.make("CartPole-v0")
        cls.continuous_env = gym.make("Pendulum-v0")

    def test_get_replay_buffer(self):
        # Replay Buffer
        rb = get_replay_buffer(self.off_policy_agent, self.discrete_env)
        self.assertTrue(isinstance(rb, ReplayBuffer))

        # Prioritized Replay Buffer
        rb = get_replay_buffer(self.off_policy_agent,
                               self.discrete_env,
                               use_prioritized_rb=True)
        self.assertTrue(isinstance(rb, PrioritizedReplayBuffer))


if __name__ == '__main__':
    unittest.main()
