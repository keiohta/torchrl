import unittest

from torchrl.algos import DQN

from tests.algos.common import CommonOffPolDiscreteAlgos


class TestDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            device='cpu',
            batch_size=cls.batch_size,
            epsilon=1.,
        )


if __name__ == '__main__':
    unittest.main()
