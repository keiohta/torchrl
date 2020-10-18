import torch
from torch.distributions import Normal
from torch.distributions.independent import Independent
import torch.nn as nn


class GaussianMLP(nn.Module):
    def __init__(self,
                 state_shape,
                 action_dim,
                 units=(8, 8),
                 hidden_nonlinearity=nn.Tanh,
                 w_init=nn.init.xavier_normal_,
                 b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1.e-6,
                 max_std=None,
                 std_parameterization='exp',
                 normal_distribution_cls=Normal):
        super(GaussianMLP, self).__init__()

        self._state_shape = state_shape
        self._action_dim = action_dim
        self._std_parameterization = std_parameterization
        self._norm_dist_class = normal_distribution_cls

        self._mean_module = nn.Sequential()
        in_dim = state_shape
        for idx, unit in enumerate(units):
            linear_layer = nn.Linear(in_dim, unit)
            w_init(linear_layer.weight)
            b_init(linear_layer.bias)

            self._mean_module.add_module(f'linear_{idx}', linear_layer)
            if hidden_nonlinearity:
                self._mean_module.add_module('non_linear_{idx}',
                                             hidden_nonlinearity())

            in_dim = unit

        linear_layer = nn.Linear(in_dim, action_dim)
        w_init(linear_layer.weight)
        b_init(linear_layer.bias)
        self._mean_module.add_module('out', linear_layer)

        init_std_param = torch.Tensor([init_std]).log()
        if learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()

    def _get_mean_and_log_std(self, *inputs):
        assert len(inputs) == 1
        mean = self._mean_module(*inputs)

        broadcast_shape = list(inputs[0].shape[:-1]) + [self._action_dim]
        uncentered_log_std = torch.zeros(*broadcast_shape) + self._init_std

        return mean, uncentered_log_std

    def forward(self, *inputs):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        dist = self._norm_dist_class(mean, std)

        # Makes it so that a sample from the distribution is treated as a
        # single sample and not dist.batch_shape samples.
        dist = Independent(dist, 1)

        return dist
