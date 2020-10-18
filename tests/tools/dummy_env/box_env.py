"""
MIT License

Copyright (c) 2019 Reinforcement Learning Working Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Dummy akro.Box environment for testing purpose.
import akro
import numpy as np

from tests.tools.dummy_env import DummyEnv


class DummyBoxEnv(DummyEnv):
    """A dummy gym.spaces.Box environment.

    Args:
        random (bool): If observations are randomly generated or not.
        obs_dim (iterable): Observation space dimension.
        action_dim (iterable): Action space dimension.

    """
    def __init__(self, random=True, obs_dim=(4, ), action_dim=(2, )):
        super().__init__(random, obs_dim, action_dim)

    @property
    def observation_space(self):
        """Return an observation space.

        Returns:
            gym.spaces: The observation space of the environment.

        """
        return akro.Box(low=-1, high=1, shape=self._obs_dim, dtype=np.float32)

    @property
    def action_space(self):
        """Return an action space.

        Returns:
            gym.spaces: The action space of the environment.

        """
        return akro.Box(low=-5.0,
                        high=5.0,
                        shape=self._action_dim,
                        dtype=np.float32)

    def reset(self):
        """Reset the environment.

        Returns:
            np.ndarray: The observation obtained after reset.

        """
        return np.ones(self._obs_dim, dtype=np.float32)

    def step(self, action):
        """Step the environment.

        Args:
            action (int): Action input.

        Returns:
            np.ndarray: Observation.
            float: Reward.
            bool: If the environment is terminated.
            dict: Environment information.

        """
        return self.observation_space.sample(), 0, False, dict(dummy='dummy')
