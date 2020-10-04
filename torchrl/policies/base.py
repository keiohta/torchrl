import torch.nn as nn


class Policy(nn.Module):
    def __init__(self,
                 name,
                 memory_capacity,
                 update_interval=1,
                 batch_size=256,
                 discount=0.99,
                 n_warmup=0,
                 max_grad=10.,
                 n_epoch=1):
        super().__init__()

        self.policy_name = name
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.discount = discount
        self.n_warmup = n_warmup
        self.n_epoch = n_epoch
        self.max_grad = max_grad
        self.memory_capacity = memory_capacity

    def get_action(self, observation, test=False):
        raise NotImplementedError

    @staticmethod
    def get_argument(parser=None):
        import argparse
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--n-warmup', type=int, default=int(1e4))
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--gpu', type=int, default=0, help='GPU id')
        return parser


class OffPolicyAgent(Policy):
    """Base class for off-policy agents"""
    def __init__(self, memory_capacity, **kwargs):
        super().__init__(memory_capacity=memory_capacity, **kwargs)
