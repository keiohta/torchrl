import unittest

from torchrl.algos import GAIL
from tests.algos.common import CommonIRLAlgos


class TestGAIL(CommonIRLAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete = GAIL(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            device='cpu',
            name='GAIL')
        cls.irl_continuous = GAIL(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            device='cpu',
            name='GAIL')


if __name__ == '__main__':
    unittest.main()
