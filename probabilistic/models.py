import torch

from globals import TORCH_DEVICE


class WeightModel:
    def __init__(self, prior_mean: torch.Tensor, prior_variance: torch.Tensor,
                 ll_variance: torch.Tensor, low: torch.Tensor = torch.tensor(-1), high: torch.Tensor = torch.tensor(1)):
        # These are needed only for gaussian
        self.prior_mean = prior_mean.to(TORCH_DEVICE)
        self.prior_variance = prior_variance.to(TORCH_DEVICE)
        # These are needed only for uniform prior
        self.a = low.to(TORCH_DEVICE)
        self.b = high.to(TORCH_DEVICE)
        # This is needed for when using the weights in a nn
        self.ll_var = ll_variance.to(TORCH_DEVICE)

    def log_gaussian_prior(self, q):
        # this assumes the same variance for every dimension
        return - (1 / (2 * self.prior_variance)) * (torch.t(q - self.prior_mean) @ (q - self.prior_mean))

    def log_gaussian_likelihood(self, dset_x: torch.Tensor, dset_y: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
        # at this point the weights from the last Markov Chain sample are set, do a forward pass
        pred_y = net(dset_x)
        # since pred_y is already softmaxed, calculate the cross entropy loss for this batch
        batch_diff = torch.nn.functional.cross_entropy(pred_y, dset_y, reduction='none')
        # compute the likelihood which is ~ N(y | pred_y, ll_sigma)
        # again, this assumes the same variance for every dimension
        return (- 0.5 * (batch_diff ** 2) / self.ll_var)

    def uniform_prior(self, q):
        volume = (self.b - self.a) ** q.shape[0]
        return torch.tensor(1 / volume)
