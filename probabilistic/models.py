import torch


class WeightModel:
    def __init__(self, prior_mean: torch.Tensor = None, prior_variance: torch.Tensor = None,
                 ll_var: torch.Tensor = torch.tensor(1.0), low: torch.Tensor = None, high: torch.Tensor = None):
        # These are needed only for gaussian
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        # These are needed only for uniform prior
        self.a = low
        self.b = high
        # This is needed for when using the weights in a nn
        self.ll_var = ll_var

    # this works regardless of the q dimension
    def gaussian_prior(self, q):
        return torch.exp(- 0.5 * (torch.t(q - self.prior_mean) @ torch.inverse(self.prior_variance) @ (q - self.prior_mean)))

    def log_gaussian_prior(self, q):
        return - (1 / (2 * self.prior_variance)) * (torch.t(q - self.prior_mean) @ (q - self.prior_mean))

    def uniform_prior(self, q):
        volume = (self.b - self.a) ** q.shape[0]
        return torch.tensor(1 / volume)

    # In this case, we want q to be the parameters of the family of distributions
    # we assume w to be part of, i.e. the bayes rule would be:
    # log p(theta | W) ~ p(W | theta) * p(theta)
    # where W represents the set of weights of the neural network
    def parameterized_gaussian_log_likelihood(self, q, x):
        return torch.exp(- 0.5 * ((x - q[0]) ** 2) / q[1])

    def nn_empirical_gaussian_likelihood(self, dset_x: torch.Tensor, dset_y: torch.Tensor, net: torch.nn.Module):
        # at this point the weights from the last Markov Chain sample are set, do a forward pass
        pred_y = net(dset_x)
        # since pred_y is already softmaxed, calculate the cross entropy loss for this batch
        batch_diff = torch.nn.functional.cross_entropy(pred_y, dset_y, reduction='none')
        # compute the likelihood which is ~ N(y | pred_y, sigma)
        return (- 0.5 * (batch_diff ** 2) / self.ll_var)
