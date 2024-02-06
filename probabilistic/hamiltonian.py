from typing import Callable, Union

import torch
from torch.utils.data import DataLoader, Dataset

from globals import TORCH_DEVICE

from .bnn import BNN


class Hamiltonian:
    def __init__(self, prior: Callable[[torch.Tensor], torch.Tensor], likelihood: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 m_variances: torch.Tensor, dset: Dataset, net: BNN = None, batch_size: int = 1000) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.m_variances = m_variances.to(TORCH_DEVICE)
        self.dset = DataLoader(dset, batch_size=batch_size, shuffle=True)
        self.net = net
        self.U = self._potential_bnn
        self.K = self._kinetic

    def grad_u(self, q: torch.Tensor):
        return torch.autograd.grad(outputs=self.U(q), inputs=q, retain_graph=True)[0]

    def hamiltonian(self, q, p):
        return self.U(q) + self.K(p)

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, lr: float) -> torch.Tensor:
        return param - lr * grad

    def energy_delta(self, q: torch.Tensor, p: torch.Tensor, current_q: torch.Tensor, current_p: torch.Tensor) -> torch.Tensor:
        return torch.exp(- self.hamiltonian(q, p) + self.hamiltonian(current_q, current_p))

    def rebatch(self, batch_size: int):
        self.dset = DataLoader(self.dset.dataset, batch_size=batch_size, shuffle=True)

    def _potential_bnn(self, q: torch.Tensor) -> torch.Tensor:
        # update the parameters (weights and biases) of the network
        self.net.init_params(q)
        batch = next(iter(self.dset)) # correct because data is shuffled anyways
        xs, ys = batch[0].to(TORCH_DEVICE), batch[1].to(TORCH_DEVICE)

        return - self.prior(q) - self.likelihood(xs, ys, self.net).sum()

    def _kinetic(self, p: torch.Tensor) -> torch.Tensor:
        return (p ** 2 / (2 * self.m_variances)).sum()

class HyperparamsHMC:
    def __init__(self, num_epochs: int, num_burnin_epochs: int, lf_step: float, steps_per_epoch: int = -1,
                 batch_size: int = 1, batches_per_epoch: int = -1, gradient_norm_bound: float = -1, dp_sigma: float = 1.0):
        self.num_epochs = num_epochs
        self.num_burnin_epochs = num_burnin_epochs
        self.lf_step = lf_step
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.gradient_norm_bound = gradient_norm_bound
        self.sigma = dp_sigma
