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
# Wrapper class that converts gym.Env into GymEnv.

import copy
import warnings

import akro
import gym


class GymEnv:
    def __new__(cls, *args, **kwargs):
        """Returns environment specific wrapper based on input environment type.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
             garage.envs.bullet.BulletEnv: if the environment is a bullet-based
                environment. Else returns a garage.envs.GymEnv
        """
        # pylint: disable=import-outside-toplevel
        # Determine if the input env is a bullet-based gym environment
        env = None
        if 'env' in kwargs:  # env passed as a keyword arg
            env = kwargs['env']
        elif len(args) >= 1:
            # env passed as a positional arg
            env = args[0]

        if isinstance(env, gym.Env):
            if env.spec and hasattr(env.spec,
                                    'id') and env.spec.id.find('Bullet') >= 0:
                from garage.envs.bullet import BulletEnv
                return BulletEnv(*args, **kwargs)
        elif isinstance(env, str):
            if 'Bullet' in env:
                from garage.envs.bullet import BulletEnv
                return BulletEnv(*args, **kwargs)

        return super(GymEnv, cls).__new__(cls)

    def __init__(self, env, is_image=False, max_episode_length=None):
        """Initializes a GymEnv.

        Note that if `env` and `env_name` are passed in at the same time,
        `env` will be wrapped.

        Args:
            env (gym.Env or str): An gym.Env Or a name of the gym environment
                to be created.
            is_image (bool): True if observations contain pixel values,
                false otherwise. Setting this to true converts a gym.Spaces.Box
                obs space to an akro.Image and normalizes pixel values.
            max_episode_length (int): The maximum steps allowed for an episode.

        Raises:
            ValueError: if `env` neither a gym.Env object nor a string.
            RuntimeError: if the environment is wrapped by a TimeLimit and its
                max_episode_steps is not equal to its spec's time limit value.
        """
        self._env = None
        if isinstance(env, str):
            self._env = gym.make(env)
        elif isinstance(env, gym.Env):
            self._env = env
        else:
            raise ValueError('GymEnv can take env as either a string, '
                             'or an Gym environment, but got type {} '
                             'instead.'.format(type(env)))

        self._render_modes = self._env.metadata['render.modes']

        self._step_cnt = None
        self._visualize = False

        self._action_space = akro.from_gym(self._env.action_space)
        self._observation_space = akro.from_gym(self._env.observation_space,
                                                is_image=is_image)

    def reset(self):
        first_states = self._env.reset()
        self._step_cnt = 0

        return first_states, dict()

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self._render_modes
