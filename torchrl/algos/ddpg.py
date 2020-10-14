from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.misc import update_network_variables
from torchrl.policies import OffPolicyAgent


class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, max_action, units=(400, 300)):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            OrderedDict([('l1', nn.Linear(state_shape[0], units[0])),
                         ('relu1', nn.ReLU()),
                         ('l2', nn.Linear(units[0], units[1])),
                         ('relu2', nn.ReLU()),
                         ('l3', nn.Linear(units[1], action_dim))]))

        self.tanh = nn.Tanh()
        self.max_action = max_action

    def forward(self, inputs):
        features = self.net(inputs)
        action = self.max_action * self.tanh(features)

        return action


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim, units=(400, 300)):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            OrderedDict([('l1', nn.Linear(state_shape[0] + action_dim,
                                          units[0])), ('relu1', nn.ReLU()),
                         ('l2', nn.Linear(units[0], units[1])),
                         ('relu2', nn.ReLU()), ('l3', nn.Linear(units[1],
                                                                1))]))

    def forward(self, states, actions):
        features = torch.cat((states, actions), dim=1)
        features = self.net(features)

        return features


class DDPG(OffPolicyAgent):
    def __init__(self,
                 state_shape,
                 action_dim,
                 device,
                 name='DDPG',
                 max_action=1.,
                 lr_actor=0.001,
                 lr_critic=0.001,
                 actor_units=(400, 300),
                 critic_units=(400, 300),
                 sigma=0.1,
                 tau=0.005,
                 n_warmup=int(1e4),
                 memory_capacity=int(1e6),
                 **kwargs):
        super().__init__(name=name,
                         memory_capacity=memory_capacity,
                         n_warmup=n_warmup,
                         **kwargs)

        self.device = device

        # define and initialize Actor network (policy network)
        self._setup_actor(state_shape, action_dim, max_action, actor_units,
                          device, lr_actor)
        # define and initialize Critic network (Q-network)
        self._setup_critic(state_shape, action_dim, critic_units, device,
                           lr_critic)

        # set hyper-parameters
        self.sigma = sigma
        self.tau = tau

    def _setup_actor(self, state_shape, action_dim, max_action, actor_units,
                     device, lr_actor):
        self.actor = Actor(state_shape, action_dim, max_action,
                           actor_units).to(device)
        self.actor_target = Actor(state_shape, action_dim, max_action,
                                  actor_units).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        update_network_variables(self.actor_target, self.actor, tau=1.)

    def _setup_critic(self, state_shape, action_dim, critic_units, device,
                      lr_critic):
        self.critic = Critic(state_shape, action_dim, critic_units).to(device)
        self.critic_target = Critic(state_shape, action_dim,
                                    critic_units).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        update_network_variables(self.critic_target, self.critic, tau=1.)

    def get_action(self, state, test=False, tensor=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        is_single_state = len(state.shape) == 1
        state = state.expand([1, state.shape[0]]).type(
            torch.float32) if is_single_state else state
        action = self._get_action_body(state, self.sigma * (1. - test),
                                       self.actor.max_action)

        return action.cpu().detach().numpy()[0] if is_single_state else action

    def _get_action_body(self, state, sigma, max_action):
        action = self.actor(state)
        if sigma > 0.:
            action += torch.normal(mean=0., std=sigma, size=action.shape)
        return torch.clamp(action, -max_action, max_action)

    def train(self,
              states,
              actions,
              next_states,
              rewards,
              done,
              weights=None,
              wandb_dict=None):
        if weights is None:
            weights = torch.ones_like(rewards)
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done, weights)

        if wandb_dict is not None:
            if actor_loss is not None:
                wandb_dict['actor_loss'] = actor_loss
            wandb_dict['ciritc_loss', critic_loss]

        return td_errors

    def _train_body(self, states, actions, next_states, rewards, done,
                    weights):
        states, actions = states.to(self.device), actions.to(self.device)
        td_errors = self._compute_td_error_body(states, actions, next_states,
                                                rewards, done)
        critic_loss = torch.mean(td_errors**2)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        next_action = self.actor(states)
        actor_loss = -torch.mean(self.critic(states, next_action))

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update target networks
        update_network_variables(self.critic_target, self.critic, self.tau)
        update_network_variables(self.actor_target, self.actor, self.tau)

        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_errors = self._compute_td_error_body(states, actions, next_states,
                                                rewards, dones)
        return np.abs(np.ravel(td_errors.cpu().detach().numpy()))

    def _compute_td_error_body(self, states, actions, next_states, rewards,
                               dones):
        not_dones = 1. - dones.type(dtype=torch.float32)
        next_act_target = self.actor_target(next_states)
        next_q_target = self.critic_target(next_states, next_act_target)
        target_q = rewards + not_dones * self.discount * next_q_target
        current_q = self.critic(states, actions)
        td_errors = target_q.detach() - current_q
        return td_errors
