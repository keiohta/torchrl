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
# Dummy environment for testing purpose.
import gym


class DummyEnv(gym.Env):
    """Base dummy environment.

    Args:
        random (bool): If observations are randomly generated or not.
        obs_dim (iterable): Observation space dimension.
        action_dim (iterable): Action space dimension.

    """
    def __init__(self, random, obs_dim=(4, ), action_dim=(2, )):
        self.random = random
        self.state = None
        self._obs_dim = obs_dim
        self._action_dim = action_dim

    @property
    def observation_space(self):
        """Return an observation space."""
        raise NotImplementedError

    @property
    def action_space(self):
        """Return an action space."""
        raise NotImplementedError

    def reset(self):
        """Reset the environment."""
        raise NotImplementedError

    def step(self, action):
        """Step the environment.

        Args:
            action (int): Action input.

        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Render.

        Args:
            mode (str): Render mode.

        """
