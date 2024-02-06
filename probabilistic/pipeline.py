from typing import List, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader

from globals import TORCH_DEVICE
from probabilistic.bnn import BNN
from probabilistic.hamiltonian import Hamiltonian, HyperparamsHMC


def init_position_and_momentum(current_q: torch.Tensor, hamiltonian: Hamiltonian, hyperparams: HyperparamsHMC, dp: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # important to start recording the gradient
    q = current_q.clone().detach().requires_grad_(True)
    p = torch.normal(mean=0, std=1, size=(current_q.shape[0],)).requires_grad_(True)
    current_p = p.clone().detach().to(TORCH_DEVICE)
    q, p = q.to(TORCH_DEVICE), p.to(TORCH_DEVICE)
    grad_q = hamiltonian.grad_u(q)
    if dp:
        # clip the gradient norm (first term) and add noise (second term)
        grad_q /= max(1, torch.norm(grad_q) / hyperparams.gradient_norm_bound)
        grad_q += torch.normal(mean=0, std=hyperparams.sigma * hyperparams.gradient_norm_bound, size=(grad_q.shape[0],))
    p = hamiltonian.update_param(p, grad_q, hyperparams.lf_step / 2)
    q = hamiltonian.update_param(q, p, - hyperparams.lf_step)
    return q, p, current_p

def hmc(hamiltonian: Hamiltonian, hyperparams: HyperparamsHMC, dp: bool = False) -> List[torch.Tensor]:
    torch.autograd.set_detect_anomaly(True)

    net = hamiltonian.net
    total_num_params = net.get_num_params()

    # init params
    samples = []
    # q is now a vector of flattened weights
    current_q = torch.normal(mean=0, std=1, size=(total_num_params,)).to(TORCH_DEVICE)
    for epoch in range(hyperparams.num_epochs):
        print(f'Epoch {epoch}...')

        q, p, current_p = init_position_and_momentum(current_q, hamiltonian, hyperparams, dp)

        # TODO: might be worth generalizing this to an enumeration of the different types of inner loops
        if dp:
            # vanilla hmc
            q, p = integration_step_dp(q, p, hamiltonian, hyperparams)
        else:
            # dp hmc
            q, p = integration_step(q, p, hamiltonian, hyperparams)

        states_energy_diff = hamiltonian.energy_delta(q, p, current_q, current_p)
        acceptance_probability = min(1, float(states_energy_diff))
        print(f'Acceptance probability: {acceptance_probability}')
        if float(torch.rand(1)) < acceptance_probability:
            current_q = q.clone().detach()
            current_p = p.clone().detach()
            if epoch > hyperparams.num_burnin_epochs:
                samples.append(current_q)
                print(f'Added sample {current_q}')

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
    clip_grad = lambda grad: grad / max(1, torch.norm(grad) / hyperparams.gradient_norm_bound)
    noisify = lambda grad: grad + torch.normal(mean=0, std=hyperparams.gradient_norm_bound * hyperparams.sigma, size=(grad.shape[0],))
    for _ in range(hyperparams.steps_per_epoch):
        # at each leapfrog step, sample a mini-batch of size batch_size
        hamiltonian.rebatch(1)
        # clip gradient norm and add noise for DP
        noisy_clipped_grad = noisify(clip_grad(hamiltonian.grad_u(q)))
        p = hamiltonian.update_param(p, noisy_clipped_grad, hyperparams.lf_step)
        q = hamiltonian.update_param(q, p, - hyperparams.lf_step)

    # make one last half leapfrog step
    p = hamiltonian.update_param(p, noisify(clip_grad(hamiltonian.grad_u(q))), hyperparams.lf_step / 2)
    # make the proposal symmetric
    p = -p
    return q, p



def test_hmc(net: BNN, param_samples: List[torch.Tensor], test_data: torchvision.datasets.mnist):
    net.eval()
    data_loader = DataLoader(test_data, batch_size=5000, shuffle=True)

    predictive_distribution = []
    for idx, param_sample in enumerate(param_samples):
        if idx % 7 == 0:
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
                batch_softmax = net(batch_data_test)
                sample_predictions.append(batch_softmax)

        # flatten it into a single tensor of size test_size x labels
        predictive_distribution.append(torch.cat(sample_predictions))

    predictive_distribution = torch.stack(predictive_distribution)

    print("shape of all predictions: ", predictive_distribution.shape)

    # Get the most often encountered class for each example
    predicted_label_mean = predictive_distribution.mean(dim=0)
    print("shape of mean predictions: ", predicted_label_mean.shape)
    predicted_labels = torch.argmax(predicted_label_mean, dim=1)
    print("shape of predicted labels: ", predicted_labels.shape)
    # compute the accuracy
    accuracy = (predicted_labels == test_data.targets.to(TORCH_DEVICE)).sum().item() / len(test_data.targets)

    return accuracy
