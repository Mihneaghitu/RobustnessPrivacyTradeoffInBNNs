from typing import Callable, Union

import torch
from torch.utils.data import DataLoader, Dataset

from globals import TORCH_DEVICE

from .bnn import BNN


class Hamiltonian:
    def __init__(self, prior: Callable[[torch.Tensor], torch.Tensor], likelihood: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 m_variances: torch.Tensor, dset: Dataset, net: BNN = None) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.m_variances = m_variances.to(TORCH_DEVICE)
        self.dset = DataLoader(dset, batch_size=5000, shuffle=True)
        self.net = net
        self.U = self._potential_bnn
        self.K = self._kinetic

    def grad_u(self, q: torch.Tensor):
        return torch.autograd.grad(outputs=self.U(q), inputs=q, retain_graph=True)[0]

    def hamiltonian(self, q, p):
        return self.U(q) + self.K(p)

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, lr: float) -> torch.Tensor:
        return param - lr * grad

    def joint_canonical_distribution(self, q: torch.Tensor, p: torch.Tensor, sgn=-1):
        return torch.exp(sgn * self.hamiltonian(q, p)) # (1 / z) * exp(-H)

    def rebatch(self, batch_size: int):
        self.dset = DataLoader(self.dset.dataset, batch_size=batch_size, shuffle=True)

    def _potential_bnn(self, q: torch.Tensor) -> torch.Tensor:
        batch = next(iter(self.dset)) # correct because data is shuffled anyways
        xs, ys = batch[0].to(TORCH_DEVICE), batch[1].to(TORCH_DEVICE)

        return - self.prior(q) - self.likelihood(xs, ys, self.net).sum()

    def _kinetic(self, p: torch.Tensor) -> torch.Tensor:
        return (p ** 2 / (0.5 * self.m_variances)).sum()
