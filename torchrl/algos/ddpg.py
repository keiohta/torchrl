from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchrl.misc import update_network_variables
from torchrl.policies import OffPolicyAgent

torch.backends.cudnn.benchmark = True


class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, max_action, units=(400, 300)):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_shape[0], units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], action_dim)

        self.max_action = max_action

    def forward(self, inputs):
        features = F.relu(self.l1(inputs))
        features = F.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * torch.tanh(features)

        return action


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim, units=(400, 300)):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_shape[0] + action_dim, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], 1)

    def forward(self, states, actions):
        features = torch.cat((states, actions), dim=1)
        features = F.relu(self.l1(features))
        features = F.relu(self.l2(features))
        values = self.l3(features)

        return torch.squeeze(values, -1)


class DDPG(OffPolicyAgent):
    def __init__(self,
                 state_shape,
                 action_dim,
                 device,
                 name='DDPG',
                 max_action=1.,
                 lr_actor=1.e-4,
                 lr_critic=1.e-3,
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
        self.actor = Actor(state_shape, action_dim, max_action,
                           actor_units).to(device)
        self.actor_target = Actor(state_shape, action_dim, max_action,
                                  actor_units).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        update_network_variables(self.actor_target, self.actor, tau=1.)
        self._update_net_requires_grad(self.actor_target, False)

        # define and initialize Critic network (Q-network)
        self.critic = Critic(state_shape, action_dim, critic_units).to(device)
        self.critic_target = Critic(state_shape, action_dim,
                                    critic_units).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        update_network_variables(self.critic_target, self.critic, tau=1.)
        self._update_net_requires_grad(self.critic_target, False)

        # set hyper-parameters
        self.sigma = sigma
        self.tau = tau

    def _update_net_requires_grad(self, net, requires_grad=False):
        for p in net.parameters():
            p.requires_grad = requires_grad

    def get_action(self, state, test=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        is_single_state = len(state.shape) == 1
        state = state.expand([1, state.shape[0]]).type(
            torch.float32) if is_single_state else state
        action = self._get_action_body(state, self.sigma * (1. - test),
                                       self.actor.max_action)

        return action.detach().numpy()[0] if is_single_state else action

    def _get_action_body(self, state, sigma, max_action):
        with torch.no_grad():
            action = self.actor(state).cpu()
            if sigma > 0.:
                action.add_(torch.normal(mean=0., std=sigma,
                                         size=action.shape))
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
            wandb_dict['ciritc_loss'] = critic_loss

        return td_errors

    def _train_body(self, states, actions, next_states, rewards, done,
                    weights):
        td_errors = self._compute_td_error_body(states, actions, next_states,
                                                rewards, done)
        critic_loss = torch.mean(td_errors.pow(2))

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()

        # Freeze Q-networks to save computational effort during policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False

        next_action = self.actor(states)
        actor_loss = -torch.mean(self.critic(states, next_action))

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        # Unfreeze Q-networks for next step
        for p in self.critic.parameters():
            p.requires_grad = True

        # Update target networks
        with torch.no_grad():
            update_network_variables(self.critic_target, self.critic, self.tau)
            update_network_variables(self.actor_target, self.actor, self.tau)

        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_errors = self._compute_td_error_body(states, actions, next_states,
                                                rewards, dones)
        return np.abs(np.ravel(td_errors.cpu().detach().numpy()))

    def _compute_td_error_body(self, states, actions, next_states, rewards,
                               dones):
        with torch.no_grad():
            not_dones = torch.sub(1., dones)
            next_act_target = self.actor_target(next_states)
            next_q_target = self.critic_target(next_states, next_act_target)
            target_q = torch.add(rewards,
                                 not_dones * self.discount * next_q_target)
        current_q = self.critic(states, actions)
        td_errors = target_q - current_q
        return td_errors
