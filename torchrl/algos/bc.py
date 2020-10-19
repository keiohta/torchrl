from collections import OrderedDict

import torch.nn as nn
import torch.optim as optim

from torchrl.policies import GaussianActor


class BC(IRLPolicy):
    def __init__(self, state_shpae, action_dim, device, name='BC'):
        self.policy = GaussianActor(state_shape,
                                    action_dim,
                                    device,
                                    units=(32, 32),
                                    lr=1.e-3)
        super().__init__(n_training=1, **kwargs)
        self.policy = GaussianMLPPolicy(
            states_shape,
            act_dim,
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

    def _train_body(self, states, actions, next_states, rewards, done,
                    weights):
        states, actions = states.to(self.device), actions.to(self.device)

        action_dist, _ = self.policy(states)
        self.optim.zero_grad()
        action_samples = action_dist.rsample()
        loss = torch.mean((actions - action_samples)**2)
        loss.backward()
        self.optim.step()

        return loss
