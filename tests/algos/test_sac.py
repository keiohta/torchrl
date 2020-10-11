import unittest

from torchrl.algos import SAC
from tests.algos.common import CommonOffPolContinuousAlgos


class TestSAC(CommonOffPolContinuousAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = SAC(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            device=cls.device,
            batch_size=cls.batch_size,
        )


class TestSACAutoAlpha(CommonOffPolContinuousAlgos):
    # TODO: Skip duplicated tests called in TestSAC
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = SAC(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            device=cls.device,
            batch_size=cls.batch_size,
            auto_alpha=True,
        )


if __name__ == '__main__':
    unittest.main()
