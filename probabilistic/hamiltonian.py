from typing import Callable

import torch


class Hamiltonian:
    def __init__(self, prior: Callable[[torch.Tensor], torch.Tensor], likelihood: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 m_variances: torch.Tensor, dset: torch.Tensor) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.m_variances = m_variances
        self.dset = dset
        self.U = self._potential
        self.K = self._kinetic

    def grad_u(self, q: torch.Tensor):
        return torch.autograd.grad(outputs=self.U(q), inputs=q, retain_graph=True)[0]

    def hamiltonian(self, q, p):
        return self.U(q) + self.K(p)

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, lr: float) -> torch.Tensor:
        return param - lr * grad

    def joint_canonical_distribution(self, q: torch.Tensor, p: torch.Tensor, sgn=-1):
        return torch.exp(sgn * self.hamiltonian(q, p))

    def _potential(self, q: torch.Tensor) -> torch.Tensor:
        # ll = self.likelihood(q, self.dset)
        return - torch.log(self.prior(q)) - torch.log(self.likelihood(q, self.dset)).sum()

    def _kinetic(self, p: torch.Tensor) -> torch.Tensor:
        return (p ** 2 / (0.5 * self.m_variances)).sum()
