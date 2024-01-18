import torch


class WeightModel:
    def __init__(self, prior_mean: torch.Tensor = None, prior_variance: torch.Tensor = None,
                 low: torch.Tensor = None, high: torch.Tensor = None):
        # These are needed only for gaussian
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        # These are needed only for uniform prior
        self.a = low
        self.b = high

    def gaussian_prior(self, q):
        return torch.exp(- 0.5 * (q - self.prior_mean).T @ torch.inverse(self.prior_variance) @ (q - self.prior_mean))

    def uniform_prior(self, q):
        return torch.where((q >= self.a) & (q <= self.b), 1 / (self.b - self.a), 0)

    def gaussian_likelihood(self, q, x):
        return torch.exp(- 0.5 * ((x - q[0]) ** 2) / q[1])
