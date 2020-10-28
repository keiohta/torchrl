from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchrl.policies import IRLPolicy


class Discriminator(nn.Module):
    def __init__(self,
                 state_shape,
                 action_dim,
                 units=(32, 32),
                 enable_sn=False,
                 output_activation='sigmoid'):
        super(Discriminator, self).__init__()

        def spectral_norm_linaer(in_dim, out_dim):
            return nn.utils.spectral_norm(nn.Linear(in_dim, out_dim))

        LinearClass = spectral_norm_linaer if enable_sn else nn.Linear

        self.l1 = LinearClass(state_shape[0] + action_dim, units[0])
        self.l2 = LinearClass(units[0], units[1])
        self.l3 = LinearClass(units[1], 1)
        self.out = nn.Sigmoid()

    def forward(self, inputs):
        features = F.relu(self.l1(inputs))
        features = F.relu(self.l2(features))
        out = self.out(self.l3(features))
        return out

    def compute_reward(self, inputs):
        return torch.log(self(inputs) + 1e-8)


class GAIL(IRLPolicy):
    def __init__(self,
                 state_shape,
                 action_dim,
                 device,
                 name='GAIL',
                 units=[32, 32],
                 lr=0.001,
                 enable_sn=False,
                 **kwargs):
        super().__init__(name=name, n_training=1, **kwargs)
        self.device = device
        self.disc = Discriminator(state_shape=state_shape,
                                  action_dim=action_dim,
                                  units=units,
                                  enable_sn=enable_sn).to(self.device)
        self.optim = optim.Adam(self.disc.parameters(),
                                lr=lr,
                                betas=(0.5, 0.999))

    def train(self,
              agent_states,
              agent_acts,
              expert_states,
              expert_acts,
              wandb_dict=None,
              **kwargs):
        loss, accuracy, js_divergence = self._train_body(
            agent_states, agent_acts, expert_states, expert_acts)

        if wandb_dict is not None:
            wandb_dict['DiscriminatorLoss'] = loss
            wandb_dict['Accuracy'] = accuracy
            wandb_dict['JSdivergence'] = js_divergence

    def _compute_js_divergence(self, fake_logits, real_logits):
        m = (fake_logits + real_logits) / 2.
        return torch.mean(
            (fake_logits * torch.log(fake_logits / m + 1e-8) +
             real_logits * torch.log(real_logits / m + 1e-8)) / 2.)

    def _train_body(self, agent_states, agent_acts, expert_states,
                    expert_acts):
        epsilon = 1e-8
        real_logits = self.disc(torch.cat((expert_states, expert_acts), dim=1))
        fake_logits = self.disc(torch.cat((agent_states, agent_acts), dim=1))
        loss = -(torch.mean(torch.log(real_logits + epsilon)) +
                 torch.mean(torch.log(1. - fake_logits + epsilon)))

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()

        accuracy = (torch.mean(
            (real_logits >= 0.5).type(torch.float32)) / 2. + torch.mean(
                (fake_logits < 0.5).type(torch.float32)) / 2.)
        js_divergence = self._compute_js_divergence(fake_logits, real_logits)
        return loss, accuracy, js_divergence

    def inference(self, states, actions, next_states):
        if states.ndim == actions.ndim == 1:
            states = states.expand([1, states.shape[0]])
            actions = actions.expand([1, actions.shape[0]])
        inputs = torch.cat([states, actions], dim=1)
        return self._inference_body(inputs)

    def _inference_body(self, inputs):
        return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--enable-sn', action='store_true')
        parser.add_argument('--expert_path_dir', type=str)
        return parser
