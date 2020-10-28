from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.policies import GaussianMLPPolicy
from torchrl.policies import IRLPolicy


class SimpleNet(nn.Module):
    def __init__(self,
                 state_shape,
                 action_dim,
                 units=(32, 32),
                 enable_sn=False,
                 output_activation='sigmoid'):
        super(SimpleNet, self).__init__()
        self.l1 = Linear(state_shape[0] + action_dim, units[0])
        self.l2 = Linear(units[0], units[1])
        self.l3 = Linear(units[1], 1)
        self.out = nn.Sigmoid()

    def forward(self, inputs):
        features = F.relu(self.l1(inputs))
        features = F.relu(self.l2(features))
        out = self.out(self.l3(features))
        return out

    def compute_reward(self, inputs):
        return torch.log(self(inputs) + 1e-8)


class BC(IRLPolicy):
    def __init__(self,
                 state_shape,
                 action_dim,
                 device,
                 units=(32, 32),
                 lr=1.e-3,
                 name='BC',
                 **kwargs):
        super().__init__(name=name, n_training=1, **kwargs)
        self.policy = GaussianMLPPolicy(
            state_shape,
            action_dim,
            device=device,
            units=units,
        ).to(device)
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device

    def train(self,
              agent_states,
              agent_acts,
              expert_states,
              expert_acts,
              wandb_dict=None,
              **kwargs):
        loss = self._train_body(agent_states, agent_acts, expert_states,
                                expert_acts)

        if wandb_dict is not None:
            wandb_dict['DiscriminatorLoss'] = loss
            wandb_dict['Accuracy'] = accuracy
            wandb_dict['JSdivergence'] = js_divergence

    def _train_body(self, states, actions, next_states, rewards):
        action_dist, _ = self.policy(torch.cat((states, actions), dim=1))
        self.optim.zero_grad()
        action_samples = action_dist.rsample()
        loss = torch.mean((actions - action_samples)**2)
        loss.backward()
        self.optim.step()

        return loss

    def inference(self, states, actions, next_states):
        if states.ndim == actions.ndim == 1:
            states = states.expand([1, states.shape[0]])
            actions = actions.expand([1, actions.shape[0]])
        inputs = torch.cat([states, actions], dim=1)
        return self._inference_body(inputs)

    def _inference_body(self, inputs):
        return self.policy(inputs)[0].rsample()

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--enable-sn', action='store_true')
        parser.add_argument('--expert_path_dir', type=str)
        return parser
