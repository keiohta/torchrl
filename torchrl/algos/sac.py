import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.policies import OffPolicyAgent, GaussianActor
from torchrl.misc import update_network_variables


class CriticV(nn.Module):
    def __init__(self, state_shape, critic_units=(256, 256)):
        super(CriticV, self).__init__()

        self.base_layers = []
        in_dim = state_shape[0]
        for unit in critic_units:
            self.base_layers.append(nn.Linear(in_dim, unit))
            self.base_layers.append(nn.ReLU())
            in_dim = unit
        self.out_layer = nn.Linear(critic_units[-1], 1)

    def forward(self, states):
        features = states
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return torch.squeeze(values, dim=1)


class CriticQ(nn.Module):
    def __init__(self, state_shape, action_dim, critic_units=(256, 256)):
        super(CriticQ, self).__init__()

        self.base_layers = []
        in_dim = state_shape[0] + action_dim
        for unit in critic_units:
            self.base_layers.append(nn.Linear(in_dim, unit))
            self.base_layers.append(nn.ReLU())
            in_dim = unit
        self.out_layer = nn.Linear(critic_units[-1], 1)

    def forward(self, features):
        for idx, cur_layer in enumerate(self.base_layers):
            features = cur_layer(features)
        values = self.out_layer(features)
        return torch.squeeze(values, dim=1)


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
        self.vf_optim = optim.Adam(self.vf.parameters(), lr=lr)

    def get_action(self, state, test=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(torch.Tensor(state), test)

        return action.detach().numpy()[0] if is_single_state else action

    def _get_action_body(self, state, test):
        actions, log_pis = self.actor(state, test)
        return actions

    def train(self,
              states,
              actions,
              next_states,
              rewards,
              dones,
              weights=None):
        if weights is None:
            weights = torch.ones_like(rewards)

        td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max, logp_mean = self._train_body(
            states, actions, next_states, rewards, dones, weights)

        return td_errors

    def _train_body(self, states, actions, next_states, rewards, dones,
                    weights):
        assert len(dones.shape) == 2
        assert len(rewards.shape) == 2
        rewards = torch.squeeze(rewards, dim=1)
        dones = torch.squeeze(dones, dim=1)

        not_dones = 1. - dones.to(dtype=torch.float32)

        # Compute loss of critic Q
        features = torch.cat([states, actions], dim=1)
        current_q1 = self.qf1(features)
        current_q2 = self.qf2(features)
        next_v_target = self.vf_target(next_states)

        target_q = (rewards +
                    not_dones * self.discount * next_v_target).detach()

        td_loss_q1 = torch.mean((current_q1 - target_q).pow(2))
        td_loss_q2 = torch.mean((current_q2 - target_q).pow(2))  # Eq.7

        self._update_optim(self.qf1_optim, self.qf1, td_loss_q1, True)
        self._update_optim(self.qf2_optim, self.qf2, td_loss_q2, True)

        # Compute loss of critic V
        current_v = self.vf(states)

        # Resample actions to update V
        sample_actions, logp = self.actor(states)

        features = torch.cat([states, sample_actions], dim=1)
        sampled_current_q1 = self.qf1(features)
        sampled_current_q2 = self.qf2(features)
        current_min_q = torch.min(sampled_current_q1, sampled_current_q2)

        target_v = (current_min_q - self.alpha * logp).detach()
        td_errors = target_v - current_v
        td_loss_v = torch.mean(td_errors.pow(2))  # Eq.(5)

        # Compute loss of policy
        policy_loss = torch.mean(self.alpha * logp - current_min_q)  # Eq.(12)

        self._update_optim(self.vf_optim, self.vf, td_loss_v, True)
        update_network_variables(self.vf_target, self.vf, self.tau)
        self._update_optim(self.actor_optim, self.actor, policy_loss, True)

        # Compute loss of temperature parameter for entropy
        if self.auto_alpha:
            alpha_loss = -torch.mean(
                (self.log_alpha * (logp + self.target_alpha).detach()))

        if self.auto_alpha:
            self._update_optim(self.alpha_optim, None, alpha_loss)
            self.alpha = torch.exp(self.log_alpha)

        return td_errors, policy_loss, td_loss_v, td_loss_q1, torch.min(
            logp), torch.max(logp), torch.mean(logp)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        if isinstance(actions, torch.Tensor):
            rewards = torch.unsqueeze(rewards, -1)
            dones = torch.unsqueeze(dones, -1)
        td_errors = self._compute_td_error_body(states, actions, next_states,
                                                rewards, dones)
        return td_errors.detach().numpy()

    def _compute_td_error_body(self, states, actions, next_states, rewards,
                               dones):
        not_dones = 1. - dones.to(dtype=torch.float32)

        # Compute TD errors for Q-value func
        features = torch.cat([states, actions], dim=1)
        current_q1 = self.qf1(features)
        vf_next_target = self.vf_target(next_states)

        target_q = (rewards +
                    not_dones * self.discount * vf_next_target).detach()

        td_errors_q1 = target_q - current_q1

        return td_errors_q1

    def _update_optim(self, optim, net, loss, retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self.grad_clip is not None:
            for p in net.modules():
                nn.utils.clip_grad_norm_(p.parameters(), self.grad_clip)
        optim.step()

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--auto-alpha', action="store_true")
        return parser
