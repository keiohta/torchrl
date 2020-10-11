import numpy as np
import torch

from torchrl.distributions import Distribution


class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        self._dim = dim

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
        old_std = torch.exp(old_log_stds)
        new_std = torch.exp(new_log_stds)

        numerator = (torch.square(old_means - new_means) +
                     torch.square(old_std) - torch.square(new_std))
        denominator = 2 * torch.square(new_std) + 1e-8
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
        assert means.shape == log_stds.shape
        zs = (x - means) / torch.exp(log_stds)
        return (-torch.sum(log_stds, axis=-1) -
                0.5 * torch.sum(torch.square(zs), axis=-1) -
                0.5 * self.dim * torch.log(torch.tensor(2 * np.pi)))

    def sample(self, param):
        means = param["mean"]
        log_stds = param["log_std"]
        # reparameterization
        return means + torch.normal(mean=0.0, std=1.0,
                                    size=means.shape) * torch.exp(log_stds)

    def entropy(self, param):
        log_stds = param["log_std"]
        return torch.sum(log_stds + torch.log(torch.sqrt(2 * np.pi * np.e)),
                         axis=-1)
