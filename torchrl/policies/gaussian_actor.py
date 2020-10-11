import numpy as np
import torch
import torch.nn as nn

from torchrl.distributions import DiagonalGaussian


class GaussianActor(nn.Module):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self,
                 state_shape,
                 action_dim,
                 max_action,
                 units=[256, 256],
                 hidden_activation="relu",
                 fix_std=False,
                 const_std=0.1,
                 state_independent_std=False,
                 squash=False):
        super(GaussianActor, self).__init__()
        self.dist = DiagonalGaussian(dim=action_dim)
        self._fix_std = fix_std
        self._const_std = const_std
        self._max_action = max_action
        self._squash = squash
        self._state_independent_std = state_independent_std

        self.l1 = nn.Linear(state_shape[0], units[0])
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(units[0], units[1])
        self.relu2 = nn.ReLU()
        self.out_mean = nn.Linear(units[1], action_dim)
        if not self._fix_std:
            if self._state_independent_std:
                self.out_log_std = torch.tensor(
                    initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                    dtype=torch.float32)
            else:
                self.out_log_std = nn.Linear(units[1], action_dim)

    def _compute_dist(self, states):
        """
        Compute multivariate normal distribution

        :param states (torch.Tensor): Inputs to neural network.
            NN outputs mean and standard deviation to compute the distribution
        :return (Dict): Multivariate normal distribution
        """
        features = self.relu1(self.l1(states))
        features = self.relu2(self.l2(features))
        mean = self.out_mean(features)
        if self._fix_std:
            log_std = torch.ones_like(mean) * torch.log(self._const_std)
        else:
            if self._state_independent_std:
                log_std = torch.tile(input=torch.unsqueeze(self.out_log_std,
                                                           axis=0),
                                     multiples=[mean.shape[0], 1])
            else:
                log_std = self.out_log_std(features)
                log_std = torch.clamp(log_std, self.LOG_SIG_CAP_MIN,
                                      self.LOG_SIG_CAP_MAX)

        return {"mean": mean, "log_std": log_std}

    def forward(self, states, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        param = self._compute_dist(states)
        if test:
            raw_actions = param["mean"]
        else:
            raw_actions = self.dist.sample(param)
        logp_pis = self.dist.log_likelihood(raw_actions, param)

        actions = raw_actions

        if self._squash:
            actions = torch.tanh(raw_actions)
            logp_pis = self._squash_correction(logp_pis, actions)

        return actions * self._max_action, logp_pis

    def compute_log_probs(self, states, actions):
        actions /= self._max_action
        param = self._compute_dist(states)
        logp_pis = self.dist.log_likelihood(actions, param)
        if self._squash:
            logp_pis = self._squash_correction(logp_pis, actions)
        return logp_pis

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)

    def _squash_correction(self, logp_pis, actions):
        diff = torch.sum(torch.log(1. - actions**2 + self.EPS), axis=1)
        return logp_pis - diff
