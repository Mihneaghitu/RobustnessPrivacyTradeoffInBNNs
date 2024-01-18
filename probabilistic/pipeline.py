from typing import List

import numpy as np
import torch

from probabilistic.hamiltonian import Hamiltonian


def init_position_and_momentum(current_q: torch.Tensor, hamiltonian: Hamiltonian, lf_step: float) -> (torch.Tensor, torch.Tensor):
    # important to start recording the gradient
    q = current_q.clone().detach().requires_grad_(True)
    p = torch.normal(mean=0, std=1, size=(2,)).requires_grad_(True)
    p = hamiltonian.update_param(p, hamiltonian.grad_u(q), lf_step / 2)
    q = q + lf_step * p

    return q, p

def hmc(hamiltonian: Hamiltonian) -> List[torch.Tensor]:
    '''
    params:
        hamiltonian: Hamiltonian object that contains the potential and kinetic energy functions
        dataset: dataset to train on, in our case a list of weights
        lf_step: leapfrog step size
        num_steps: number of leapfrog steps
    '''
    torch.autograd.set_detect_anomaly(True)

    # hyperparams
    num_epochs = 100
    num_burnin_epochs = 10
    lf_step = 0.0001
    steps_per_epoch = 100

    # init params
    samples = []
    current_q = torch.rand(2) * 2 - 1 # Uniform(-1, 1)
    for epoch in range(num_epochs):
        q, p = init_position_and_momentum(current_q, hamiltonian, lf_step)
        current_p = p
        path = []
        for _ in range(steps_per_epoch):
            p = hamiltonian.update_param(p, hamiltonian.grad_u(q), lf_step)
            q = hamiltonian.update_param(q, p, -lf_step)
            path.append(q)
        # make one last half leapfrog step
        p = hamiltonian.update_param(p, hamiltonian.grad_u(q), lf_step / 2)
        # make the proposal symmetric
        p = -p
        old_state_energy = hamiltonian.joint_canonical_distribution(current_q, current_p)
        new_state_energy = hamiltonian.joint_canonical_distribution(q, p, 1)
        acceptance_probability = min(1, old_state_energy * new_state_energy)
        if torch.rand(1)[0] < acceptance_probability:
            current_q = q.clone().detach()
            current_p = p.clone().detach()
            if epoch > num_burnin_epochs:
                samples = samples + path

    return samples