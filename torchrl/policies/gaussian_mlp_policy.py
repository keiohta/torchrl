import torch
import torch.nn as nn

from torchrl.networks import GaussianMLP


class GaussianMLPPolicy(nn.Module):
    def __init__(self,
                 state_shape,
                 action_dim,
                 device,
                 units=(8, 8),
                 hidden_nonlinearity=nn.Tanh,
                 w_init=nn.init.xavier_uniform_,
                 b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp'):
        super(GaussianMLPPolicy, self).__init__()
        self.device = device
        self.module = GaussianMLP(state_shape[0],
                                  action_dim,
                                  units=units,
                                  hidden_nonlinearity=hidden_nonlinearity,
                                  w_init=w_init,
                                  b_init=b_init,
                                  learn_std=learn_std,
                                  init_std=init_std,
                                  min_std=min_std,
                                  max_std=max_std,
                                  std_parameterization=std_parameterization)

    def forward(self, states):
        dist = self.module(states)

        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))

    def get_action(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state).float().to(self.device)
            state = state.unsqueeze(0)
            action, agent_infos = self.get_actions(state)
            return action[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, states):
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.as_tensor(states).float().to(self.device)
            dist, info = self.forward(states)
            return dist.sample().cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_dist(self, states):
        with torch.no_grad():
            dist, _ = self._compute_dist(states)

        return dist
