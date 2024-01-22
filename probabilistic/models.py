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

    # this works regardless of the q dimension
    def gaussian_prior(self, q):
        return torch.exp(- 0.5 * (q - self.prior_mean).T @ torch.inverse(self.prior_variance) @ (q - self.prior_mean))

    def uniform_prior(self, q):
        volume = (self.b - self.a) ** q.shape[0]
        return torch.tensor(1 / volume)

    # In this case, we want q to be the parameters of the family of distributions
    # we assume w to be part of, i.e. the bayes rule would be:
    # log p(theta | W) ~ p(W | theta) * p(theta)
    # where W represents the set of weights of the neural network
    def parameterized_gaussian_likelihood(self, q, x):
        return torch.exp(- 0.5 * ((x - q[0]) ** 2) / q[1])

    def nn_empirical_gaussian_likelihood(self, q, x):
        # TODO: implement this with a neural network
        pass