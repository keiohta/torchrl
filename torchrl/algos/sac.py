from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchrl.policies import OffPolicyAgent, GaussianActor
from torchrl.misc import update_network_variables

torch.backends.cudnn.benchmark = True


class CriticV(nn.Module):
    def __init__(self, state_shape, critic_units=(256, 256)):
        super(CriticV, self).__init__()

        self.l1 = nn.Linear(state_shape[0], critic_units[0])
        self.l2 = nn.Linear(critic_units[0], critic_units[1])
        self.l3 = nn.Linear(critic_units[1], 1)

    def forward(self, states):
        features = F.relu(self.l1(states))
        features = F.relu(self.l2(features))
        values = self.l3(features)
        return torch.squeeze(values, -1)


class CriticQ(nn.Module):
    def __init__(self, state_shape, action_dim, critic_units=(256, 256)):
        super(CriticQ, self).__init__()

        self.l1 = nn.Linear(state_shape[0] + action_dim, critic_units[0])
        self.l2 = nn.Linear(critic_units[0], critic_units[1])
        self.l3 = nn.Linear(critic_units[1], 1)

    def forward(self, features):
        features = F.relu(self.l1(features))
        features = F.relu(self.l2(features))
        values = self.l3(features)
        return torch.squeeze(values, -1)


class SAC(OffPolicyAgent):
    def __init__(self,
                 state_shape,
                 action_dim,
                 device,
                 name="SAC",
                 max_action=1.,
                 lr=3e-4,
                 actor_units=(256, 256),
                 critic_units=(256, 256),
                 tau=5e-3,
                 alpha=.2,
                 auto_alpha=False,
                 n_warmup=int(1e4),
                 memory_capacity=int(1e6),
                 grad_clip=None,
                 **kwargs):
        super().__init__(name=name,
                         memory_capacity=memory_capacity,
                         n_warmup=n_warmup,
                         **kwargs)

        self.device = device
        self._setup_actor(state_shape, action_dim, actor_units, lr, max_action)
        self._setup_critic_v(state_shape, critic_units, lr)
        self._setup_critic_q(state_shape, action_dim, critic_units, lr)

        # Set hyper-parameters
        self.tau = tau
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = torch.zeros(1,
                                         requires_grad=True,
                                         device=self.device)
            self.alpha = torch.exp(self.log_alpha)
            self.target_alpha = -action_dim
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        self.state_ndim = len(state_shape)
        self.grad_clip = grad_clip

    def _setup_actor(self,
                     state_shape,
                     action_dim,
                     actor_units,
                     lr,
                     max_action=1.):
        self.actor = GaussianActor(state_shape,
                                   action_dim,
                                   self.device,
                                   max_action,
                                   squash=True,
                                   units=actor_units).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

    def _setup_critic_q(self, state_shape, action_dim, critic_units, lr):
        self.qf1 = CriticQ(state_shape, action_dim,
                           critic_units).to(self.device)
        self.qf2 = CriticQ(state_shape, action_dim,
                           critic_units).to(self.device)
        self.qf1_optim = optim.Adam(self.qf1.parameters(), lr=lr)
        self.qf2_optim = optim.Adam(self.qf2.parameters(), lr=lr)

    def _setup_critic_v(self, state_shape, critic_units, lr):
        self.vf = CriticV(state_shape, critic_units).to(self.device)
        self.vf_target = CriticV(state_shape, critic_units).to(self.device)
        update_network_variables(self.vf_target, self.vf, tau=1.)
        self._update_net_requires_grad(self.vf_target)
        self.vf_optim = optim.Adam(self.vf.parameters(), lr=lr)

    def _update_net_requires_grad(self, net, requires_grad=False):
        for p in net.parameters():
            p.requires_grad = requires_grad

    def get_action(self, state, test=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        is_single_state = len(state.shape) == self.state_ndim

        state = state.expand(1, state.shape[0]).type(
            torch.FloatTensor) if is_single_state else state
        state = state.to(self.device)
        action = self._get_action_body(state, test)

        return action.cpu().detach().numpy()[0] if is_single_state else action

    def _get_action_body(self, state, test):
        actions, log_pis = self.actor(state, test)
        return actions

    def train(self,
              states,
              actions,
              next_states,
              rewards,
              dones,
              weights=None,
              wandb_dict=None):
        td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max, logp_mean = self._train_body(
            states, actions, next_states, rewards, dones)

        # Log to wandb if set
        if wandb_dict is not None:
            wandb_dict['actor_loss'] = actor_loss
            wandb_dict['critic_V_loss'] = vf_loss
            wandb_dict['critic_Q_loss'] = qf_loss
            wandb_dict['logp_min'] = logp_min
            wandb_dict['logp_max'] = logp_max
            wandb_dict['logp_mean'] = logp_mean
            if self.auto_alpha:
                wandb_dict['log_ent'] = log_alpha
                wandb_dict['logp_mean+target'] = logp_mean + self.target_alpha
            wandb_dict['ent'] = self.alpha

        return td_errors

    def _train_body(self, states, actions, next_states, rewards, dones):
        rewards = torch.squeeze(rewards, dim=1)
        dones = torch.squeeze(dones, dim=1)

        # Compute loss of critic Q
        features = torch.cat([states, actions], dim=1)
        current_q1 = self.qf1(features)
        current_q2 = self.qf2(features)
        with torch.no_grad():
            not_dones = 1. - dones
            next_v_target = self.vf_target(next_states)
            target_q = rewards + not_dones * self.discount * next_v_target

        td_loss_q1 = torch.mean((current_q1 - target_q).pow(2))
        td_loss_q2 = torch.mean((current_q2 - target_q).pow(2))  # Eq.7

        self._update_optim(self.qf1_optim, self.qf1, td_loss_q1)
        self._update_optim(self.qf2_optim, self.qf2, td_loss_q2)

        self._update_net_requires_grad(self.qf1, requires_grad=False)
        self._update_net_requires_grad(self.qf2, requires_grad=False)

        # Compute loss of critic V
        current_v = self.vf(states)

        # Resample actions to update V
        sample_actions, logp = self.actor(states)

        features = torch.cat([states, sample_actions], dim=1)
        sampled_current_q1 = self.qf1(features)
        sampled_current_q2 = self.qf2(features)
        current_min_q = torch.min(sampled_current_q1, sampled_current_q2)

        with torch.no_grad():
            target_v = (current_min_q - self.alpha * logp)
        td_errors = target_v - current_v
        td_loss_v = torch.mean(td_errors.pow(2))  # Eq.(5)

        # Compute loss of policy
        policy_loss = torch.mean(self.alpha * logp - current_min_q)  # Eq.(12)

        self._update_optim(self.vf_optim, self.vf, td_loss_v)
        with torch.no_grad():
            update_network_variables(self.vf_target, self.vf, self.tau)
        self._update_optim(self.actor_optim, self.actor, policy_loss)

        self._update_net_requires_grad(self.qf1, requires_grad=True)
        self._update_net_requires_grad(self.qf2, requires_grad=True)

        # Compute loss of temperature parameter for entropy
        if self.auto_alpha:
            alpha_loss = -torch.mean(
                (self.log_alpha * (logp + self.target_alpha).detach()))
            self._update_optim(self.alpha_optim, None, alpha_loss)
            self.alpha = torch.jit.script(torch.exp(self.log_alpha))

        return td_errors, policy_loss, td_loss_v, td_loss_q1, torch.min(
            logp), torch.max(logp), torch.mean(logp)

    def _update_optim(self, optim, net, loss):
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--auto-alpha', action="store_true")
        return parser
