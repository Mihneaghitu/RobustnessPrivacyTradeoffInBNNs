from typing import List

import torch

from probabilistic.bnn import BNN
from probabilistic.hamiltonian import Hamiltonian


class HyperparamsHMC:
    def __init__(self, num_epochs: int, num_burnin_epochs: int, lf_step: float, steps_per_epoch: int = -1,
                 batch_size: int = -1, batches_per_epoch: int = -1, gradient_norm_bound: float = -1):
        self.num_epochs = num_epochs
        self.num_burnin_epochs = num_burnin_epochs
        self.lf_step = lf_step
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.gradient_norm_bound = gradient_norm_bound


def init_position_and_momentum(current_q: torch.Tensor, hamiltonian: Hamiltonian, lf_step: float) -> (torch.Tensor, torch.Tensor):
    # important to start recording the gradient
    q = current_q.clone().detach().requires_grad_(True)
    p = torch.normal(mean=0, std=1, size=(current_q.shape[0],)).requires_grad_(True)
    # TODO: check if this is correct
    p = hamiltonian.update_param(p, hamiltonian.grad_u(q), lf_step / 2)
    q = q + lf_step * p

    return q, p

# def hmc(hamiltonian: Hamiltonian, hyperparams: HyperparamsHMC) -> List[torch.Tensor]:
#     torch.autograd.set_detect_anomaly(True)
#     # init params
#     samples = []
#     current_q = torch.rand(2) * 2 - 1 # Uniform(-1, 1)
#     for epoch in range(hyperparams.num_epochs):
#         q, p = init_position_and_momentum(current_q, hamiltonian, lf_step)
#         current_p = p
#         path = []
#         for _ in range(steps_per_epoch):
#             p = hamiltonian.update_param(p, hamiltonian.grad_u(q), lf_step)
#             q = hamiltonian.update_param(q, p, -lf_step)
#             path.append(q)
#         # make one last half leapfrog step
#         p = hamiltonian.update_param(p, hamiltonian.grad_u(q), lf_step / 2)
#         # make the proposal symmetric
#         p = -p
#         old_state_energy = hamiltonian.joint_canonical_distribution(current_q, current_p)
#         new_state_energy = hamiltonian.joint_canonical_distribution(q, p, 1)
#         acceptance_probability = min(1, old_state_energy * new_state_energy)
#         if torch.rand(1)[0] < acceptance_probability:
#             current_q = q.clone().detach()
#             current_p = p.clone().detach()
#             if epoch > num_burnin_epochs:
#                 samples = samples + path
#
#     return samples

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
                # TODO: check if this is correct? should the whole path be appended?
                samples = samples + path

    return samples


def dp_hmc(hamiltonian: Hamiltonian) -> List[torch.Tensor]:
    torch.autograd.set_detect_anomaly(True)

    # hyperparams
    num_epochs = 100
    num_burnin_epochs = 10
    lf_step = 0.0001
    gradient_norm_bound = 0.5
    batch_size = 32
    batches_per_epoch = 50

    # init params
    samples = []
    current_q = torch.rand(2) * 2 - 1 # Uniform(-1, 1)
    for epoch in range(num_epochs):
        q, p = init_position_and_momentum(current_q, hamiltonian, lf_step)
        current_p = p
        path = []
        for _ in range(batches_per_epoch):
            batch_avg_grad = 0
            for batch_idx in range(batch_size):
                grad_q = hamiltonian.grad_u(q)
                # clip the gradient norm
                grad_norm = grad_q / max(1, torch.norm(grad_q) / gradient_norm_bound)
                # add noise for DP
                noisy_grad_norm = grad_norm + torch.normal(mean=0, std=gradient_norm_bound, size=(grad_q.shape[0],))
                # update the average gradient
                batch_avg_grad = (batch_idx * batch_avg_grad + noisy_grad_norm) / (batch_idx + 1)
            # update the parameters
            p = hamiltonian.update_param(p, batch_avg_grad, lf_step)
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


def hmc_bnn(hamiltonian: Hamiltonian) -> List[torch.Tensor]:
    '''
    params:
        hamiltonian: Hamiltonian object that contains the potential and kinetic energy functions
        dataset: dataset to train on, in our case a list of weights
        lf_step: leapfrog step size
        num_steps: number of leapfrog steps
    '''
    torch.autograd.set_detect_anomaly(True)

    net = hamiltonian.net
    total_num_weights = net.get_num_weights()

    # hyperparams
    num_epochs = 10
    num_burnin_epochs = 2
    lf_step = .5
    steps_per_epoch = 20

    # init params
    samples = []
    # q is now a vector of flattened weights
    # TODO: check if this makes sense
    current_q = torch.rand(total_num_weights) * 2 - 1 # Uniform(-1, 1)
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
        print(acceptance_probability)
        if torch.rand(1)[0] < acceptance_probability:
            current_q = q.clone().detach()
            current_p = p.clone().detach()
            # update the weights of the network
            net.init_weights(current_q)
            if epoch > num_burnin_epochs:
                samples.append(current_q)

    return samples
