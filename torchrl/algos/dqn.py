"""
Implementation of Deep Q-Network (DQN)
Paper: Human-level control through deep reinforcement learning
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.envs import LazyFrames
from torchrl.misc import huber_loss
from torchrl.policies import OffPolicyAgent


class QFunc(nn.Module):
    def __init__(self, state_shape, action_dim, units=(32, 32)):
        super(QFunc, self).__init__()

        self.net = nn.Sequential(
            OrderedDict(([
                ('L1', nn.Linear(state_shape[0], units[0])),
                ('relu1', nn.ReLU()),
                ('L2', nn.Linear(units[0], units[1])),
                ('relu2', nn.ReLU()),
                ('L3', nn.Linear(units[1], action_dim)),
            ])))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        q_values = self.net(x)

        return q_values


class DQN(OffPolicyAgent):
    def __init__(self,
                 state_shape,
                 action_dim,
                 name='DQN',
                 q_fn=None,
                 loss_fn=None,
                 optimizer=optim.Adam,
                 units=(32, 32),
                 lr=1.e-3,
                 epsilon=1.e-1,
                 epsilon_min=None,
                 epsilon_decay_step=1e6,
                 n_warmup=1e6,
                 target_update_interval=5e3,
                 memory_capacity=int(1e6),
                 **kwargs):
        super().__init__(name=name,
                         memory_capacity=memory_capacity,
                         n_warmup=n_warmup,
                         **kwargs)

        q_fn = q_fn if q_fn is not None else QFunc
        # initialize Q function
        kwargs_dqn = {
            'state_shape': state_shape,
            'action_dim': action_dim,
            'units': units,
        }
        self.q_fn = q_fn(**kwargs_dqn)
        self.q_fn_target = q_fn(**kwargs_dqn)
        self.optimizer = optimizer(self.q_fn.parameters(), lr=lr)
        self.loss_fn = huber_loss

        self._action_dim = action_dim
        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

        # hyperparameters for trainig
        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.
        self.epsilon_min = epsilon

        self.n_update = 0
        self.target_update_interval = target_update_interval

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--enable-double-dqn', action='store_true')
        parser.add_argument('--enable-dueling-dqn', action='store_true')
        parser.add_argument('--enable-categorical-dqn', action='store_true')
        parser.add_argument('--enable-noisy-dqn', action='store_true')
        return parser

    def get_action(self, state, test=False, tensor=False):
        if isinstance(state, LazyFrames):
            state = np.array(state)
        if not tensor:
            assert isinstance(state, np.ndarray)
        is_single_input = state.ndim == self._state_ndim

        if not test and np.random.rand() < self.epsilon:
            if is_single_input:
                action = np.random.randint(self._action_dim)
            else:
                action = np.array([
                    np.random.randint(self._action_dim)
                    for _ in range(state.shape[0])
                ],
                                  dtype=np.int64)

            if tensor:
                return torch.from_numpy(action)
            else:
                return action

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_input else state
        action = self._get_action_body(torch.tensor(state))

        if tensor:
            return action
        else:
            if is_single_input:
                return action.numpy()[0]
            else:
                return action.numpy()

    def _get_action_body(self, state):
        q_values = self.q_fn(state)

        return torch.argmax(q_values, axis=1)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        td_errors, q_fn_loss = self._train(states, actions, next_states,
                                           rewards, done, weights)

        self.n_update += 1
        # update target networks
        if self.n_update % self.target_update_interval == 0:
            self.q_fn_target.load_state_dict(self.q_fn.state_dict())

        # update exploration rate
        self.epsilon = max(
            self.epsilon - self.epsilon_decay_rate * self.update_interval,
            self.epsilon_min)

        return {'td_errors': td_errors, 'q_fn_loss': q_fn_loss}

    def _train(self, states, actions, next_states, rewards, done, weights):
        current_Q, target_Q = self._compute_q_values(states, actions,
                                                     next_states, rewards,
                                                     done)
        # compute temporal difference
        td_errors = current_Q - target_Q
        # compute Huber loss
        q_fn_loss = torch.mean(
            self.loss_fn(current_Q - target_Q, delta=self.max_grad) *
            torch.from_numpy(weights))

        # optimizer model
        self.optimizer.zero_grad()
        q_fn_loss.backward()
        self.optimizer.step()

        return td_errors, q_fn_loss

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        current_Q, target_Q = self._compute_q_values(states, actions,
                                                     next_states, rewards,
                                                     dones)

        return current_Q - target_Q

    def _compute_q_values(self, states, actions, next_states, rewards, dones):
        batch_size = states.shape[0]
        not_dones = 1. - torch.from_numpy(dones)
        actions = torch.from_numpy(actions).type(torch.int32)
        rewards = torch.from_numpy(rewards)

        indices = torch.cat(
            [torch.unsqueeze(torch.arange(0, batch_size), dim=1), actions],
            dim=1).type(torch.int64)
        current_Q = torch.unsqueeze(self.q_fn(states)[list(indices.T)], dim=1)

        target_Q = rewards + not_dones * self.discount * torch.max(
            self.q_fn_target(next_states), dim=1, keepdim=True)[0]
        target_Q = target_Q.detach()  # stop computing gradient

        return current_Q, target_Q
