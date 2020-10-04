from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np


def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [1]  # space.n
    else:
        raise NotImplementedError(
            "Currently only accept types of Box or Discrete, not {}".format(
                type(space)))


def get_default_rb_dict(size, env):
    """Return default replay buffer in a dict format"""
    return {
        'size': size,
        'default_dtype': np.float32,
        'env_dict': {
            'obs': {
                'shape': get_space_size(env.observation_space)
            },
            'next_obs': {
                'shape': get_space_size(env.observation_space)
            },
            'act': {
                'shape': get_space_size(env.action_space)
            },
            'rew': {},
            'done': {}
        }
    }


def get_replay_buffer(policy,
                      env,
                      use_prioritized_rb=False,
                      use_nstep_rb=False,
                      n_step=1,
                      size=None):
    if policy is None or env is None:
        return None

    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    if size is not None:
        kwargs['size'] = size

    # TODO(sff1019): Add on-policy behaviour
    # TODO(sff1019): Add N-step prioritized

    if len(obs_shape) == 3:
        kwargs['env_dict']['obs']['dtype'] = np.ubyte
        kwargs['env_dict']['next_obs']['dtype'] = np.ubtye

    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwrgs)

    return ReplayBuffer(**kwargs)
