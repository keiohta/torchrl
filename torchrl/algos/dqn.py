"""
Implementation of Deep Q-Network (DQN)
Paper: Human-level control through deep reinforcement learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.policies import OffPolicyAgent


class QFunc(nn.Modules):
    def __init__(self, state_shape, action_dim, units=(32, 32)):
        super(self, QFunc).__init__()

        self.net = nn.Sequential(
            OrderedDict(([
                ('L1', nn.Linear(state_shape + 1, units[0])),
                ('relu1', nn.ReLU()),
                ('L2', nn.Linear(units[0], units[1])),
                ('relu2', nn.ReLU()),
                ('L3', nn.Linear(units[1], action_dim)),
            ])))

    def forward(self, x):
        q_values = self.net(x)

        return q_values


class DQN(OffPolicyAgent):
    def __init__(self,
                 q_fn=None,
                 optimizer=optim.Adam,
                 loss_fn=nn.SmoothL1Loss,
                 units=(32, 32),
                 lr=1.e-3,
                 epsilon=1.e-1,
                 epsilon_min=None,
                 epsilon_decay_step=1e6,
                 n_warmup=1e6,
                 target_update_interval=5e3,
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
        self.optimizer = optimizer(net.parameters())
        self.loss_fn = loss_fn()

        # hyperparameters for trainig
        self.epsilon = epsilon

        self.n_updates = 0
        self.target_update_interval = target_update_interval

    def train(self, states, actions, next_states, rewards, done, weights=None):
        td_errors, q_fn_loss = self._train(states, actions, next_states,
                                           rewards, done, weights)

        # update target networks
        if self.n_update % self.targtet_update_interval == 0:
            self.q_fn_target.load_state_dict(self.q_fn.state_dict())

        # update exploration rate
        self.epsilon = max(
            self.epsilon - self.epsilon_decay_rate * self.update_interval,
            self.epsilon_min)

        return td_errors

    def _train(self, states, actions, next_states, rewards, done, weights):
        current_Q, target_Q = self._compute_q_values(states, actions,
                                                     next_states, rewards,
                                                     done)
        # compute temporal difference
        td_errors = current_Q - target_Q
        # compute Huber loss
        q_fn_loss = self.loss_fn(current_Q, target_Q)

        # optimizer model
        self.optimizer.zero_grad()
        q_fn_loss.backward()
        optimizer.step()

        return td_errors, q_fn_loss

    def _compute_q_values(self):
        batch_size = states.shape[0]
        not_dones = 1. - torch.from_numpy(dones)

        indices = torch.cat(
            [torch.unsqueeze(torch.range(batch_size), dim=1), actions], dim=1)
        current_Q = self.q_fn(states).gather(1, indices)

        target_Q = rewards + not_dones * self.discount * torch.max(
            self.q_fn_target(next_states), dim=1, keepdim=True)
        target_Q = target_Q.detach()  # stop computing gradient

        return current_Q, target_Q
