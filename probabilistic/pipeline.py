from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader

from globals import TORCH_DEVICE
from probabilistic.bnn import BNN
from probabilistic.hamiltonian import Hamiltonian


class HyperparamsHMC:
    def __init__(self, num_epochs: int, num_burnin_epochs: int, lf_step: float, steps_per_epoch: int = -1,
                 batch_size: int = 1, batches_per_epoch: int = -1, gradient_norm_bound: float = -1):
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
    q, p = q.to(TORCH_DEVICE), p.to(TORCH_DEVICE)
    p = hamiltonian.update_param(p, hamiltonian.grad_u(q), lf_step / 2)
    q = q + lf_step * p

    return q, p

def hmc(hamiltonian: Hamiltonian, hyperparams: HyperparamsHMC, dp=False) -> List[torch.Tensor]:
    torch.autograd.set_detect_anomaly(True)

    net = hamiltonian.net
    total_num_params = net.get_num_params()

    # init params
    samples = []
    # q is now a vector of flattened weights
    current_q = torch.rand(total_num_params).to(TORCH_DEVICE) * 2 - 1 # Uniform(-1, 1)
    for epoch in range(hyperparams.num_epochs):
        print(f'Epoch {epoch}...')

        q, p = init_position_and_momentum(current_q, hamiltonian, hyperparams.lf_step)
        current_p = p

        # TODO: might be worth generalizing this to an enumeration of the different types of inner loops
        if dp:
            # vanilla hmc
            q, p = integration_step_dp(q, p, hamiltonian, hyperparams)
        else:
            # dp hmc
            q, p = integration_step(q, p, hamiltonian, hyperparams)

        old_state_energy = hamiltonian.joint_canonical_distribution(current_q, current_p)
        new_state_energy = hamiltonian.joint_canonical_distribution(q, p, 1)
        acceptance_probability = min(1, old_state_energy * new_state_energy)
        if torch.rand(1)[0] < acceptance_probability:
            current_q = q.clone().detach()
            current_p = p.clone().detach()
            # update the weights of the network
            net.init_params(current_q)
            if epoch > hyperparams.num_burnin_epochs:
                samples.append(current_q)

    return samples

def integration_step(q: torch.Tensor, p: torch.Tensor, hamiltonian: Hamiltonian, hyperparams: HyperparamsHMC) -> (torch.Tensor, torch.Tensor):
    for _ in range(hyperparams.steps_per_epoch):
        p = hamiltonian.update_param(p, hamiltonian.grad_u(q), hyperparams.lf_step)
        q = hamiltonian.update_param(q, p, - hyperparams.lf_step)
    # make one last half leapfrog step
    p = hamiltonian.update_param(p, hamiltonian.grad_u(q), hyperparams.lf_step / 2)
    # make the proposal symmetric
    p = -p

    return q, p

def integration_step_dp(q: torch.Tensor, p: torch.Tensor, hamiltonian: Hamiltonian, hyperparams: HyperparamsHMC) -> (torch.Tensor, torch.Tensor):
    # TODO: ask whether a subloop with a batch size like 100 is ok?
    # Also, in the current case is L (num_leapfrog_steps) equal to the batch size?
    for _ in range(hyperparams.steps_per_epoch):
        # at each leapfrog step, sample a mini-batch of size batch_size
        hamiltonian.rebatch(1)
        grad_q = hamiltonian.grad_u(q)
        # clip the gradient norm
        clipped_grad = grad_q / max(1, torch.norm(grad_q) / hyperparams.gradient_norm_bound)
        # add noise for DP
        noisy_clipped_grad = clipped_grad + torch.normal(mean=0, std=hyperparams.gradient_norm_bound, size=(grad_q.shape[0],))
        # update the parameters
        p = hamiltonian.update_param(p, noisy_clipped_grad, hyperparams.lf_step)
        q = hamiltonian.update_param(q, p, - hyperparams.lf_step)

    return q, p



def test_hmc(net: BNN, param_samples: List[torch.Tensor], test_data: torchvision.datasets.mnist):
    net.eval()
    data_loader = DataLoader(test_data, batch_size=100, shuffle=True)

    predictive_distribution = []
    for idx, param_sample in enumerate(param_samples):
        if idx % 10 == 0:
            print(f'Predicting with weight sample {idx}...')
            # mem_reserved = torch.cuda.memory_reserved(TORCH_DEVICE)
            # mem_allocated = torch.cuda.memory_allocated(TORCH_DEVICE)
            # total_mem = torch.cuda.get_device_properties(TORCH_DEVICE).total_memory
            # print(f'Memory reserved: {mem_reserved / 1e9} GB')
            # print(f'Memory allocated: {mem_allocated / 1e9} GB')
            # print(f'Total memory: {total_mem / 1e9} GB')
        # first update the net weights
        net.init_params(param_sample)
        sample_predictions = []
        with torch.no_grad():
            for data, _ in data_loader:
                batch_data_test = data.to(TORCH_DEVICE)
                batch_logits = net(batch_data_test)
                sample_predictions.append(batch_logits)

        # flatten it into a single tensor of size test_size x labels
        predictive_distribution.append(torch.cat(sample_predictions))

    predictive_distribution = torch.stack(predictive_distribution)

    print("shape of all predictions: ", predictive_distribution.shape)

    logits_mean = torch.mean(predictive_distribution, dim=0)
    # now that we have the mean logits, put everything through a softmax to get the predictive distribution
    predicted_label_mean = torch.nn.functional.softmax(logits_mean, dim=1).argmax(dim=1)
    # compute the accuracy
    accuracy = (predicted_label_mean == test_data.targets).sum().item() / len(test_data.targets)

    return accuracy