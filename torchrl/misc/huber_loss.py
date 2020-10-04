# TODO(sff1019): torchv1.6.0 does not compute Huber loss correctly, but it will be fixed.
import torch


def huber_loss(x, delta=1.):
    """
    Compute the huber loss (nn.SmoothL1Loss with delta)
    :param x (torch.Tensor): values to compute
    :param delta (float): threshold
    """
    delta = torch.ones_like(x) * delta
    lower = 0.5 * torch.square(x)
    upper = delta * (torch.abs(x) - 0.5 * delta)

    return torch.where(torch.abs(x) <= delta, lower, upper)
