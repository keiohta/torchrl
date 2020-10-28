import numpy as np
import torch

from torchrl.distributions import Distribution


class DiagonalGaussian(Distribution):
    def __init__(self, dim, device=None):
        torch.backends.cudnn.benchmark = True
        self._dim = dim
        self.device = device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def dim(self):
        return self._dim

    def kl(self, old_param, new_param):
        r"""
        Compute KL divergence of two distributions as:
            {(\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2} / (2 * \sigma_2^2) + ln(\sigma_2 / \sigma_1)

        :param old_param (Dict):
            Gaussian distribution to compare with that contains
            means: (batch_size * output_dim)
            std: (batch_size * output_dim)
        :param new_param (Dict): Same contents with old_param
        """
        old_means, old_log_stds = old_param["mean"], old_param["log_std"]
        new_means, new_log_stds = new_param["mean"], new_param["log_std"]
        old_std = torch.jit.script(torch.exp(old_log_stds))
        new_std = torch.jit.script(torch.exp(new_log_stds))

        numerator = torch.jit.script(
            torch.square(old_means - new_means) + torch.square(old_std) -
            torch.square(new_std))
        denominator = torch.jit.script(2 * torch.square(new_std) + 1e-8)
        return torch.sum(numerator / denominator + new_log_stds - old_log_stds)

    def likelihood_ratio(self, x, old_param, new_param):
        llh_new = self.log_likelihood(x, new_param)
        llh_old = self.log_likelihood(x, old_param)
        return torch.exp(llh_new - llh_old)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            TODO: write equation
        """
        means = param["mean"]
        log_stds = param["log_std"]
        zs = (x - means) / torch.exp(log_stds)

        # return _calc_log_likelihood(log_stds, zs, np.pi, self.dim)
        return (-torch.sum(log_stds, axis=-1) -
                0.5 * torch.sum(torch.square(zs), axis=-1) -
                0.5 * self.dim * torch.log(torch.tensor(2 * np.pi)))

    def sample(self, param):
        means = param["mean"]
        log_stds = param["log_std"]
        # reparameterization
        return means + torch.normal(mean=0.0, std=1.0, size=means.shape).to(
            self.device) * torch.exp(log_stds)

    def entropy(self, param):
        log_stds = param["log_std"]
        # return torch.sum(_calc_entropy(log_stds, np.pi, np.e), axis=-1)
        return torch.sum(
            torch.jit.script(log_stds +
                             torch.log(torch.sqrt(2 * np.pi * np.e))),
            axis=-1)
        # return torch.sum(log_stds + torch.log(torch.sqrt(2 * np.pi * np.e)),
        #                  axis=-1)
