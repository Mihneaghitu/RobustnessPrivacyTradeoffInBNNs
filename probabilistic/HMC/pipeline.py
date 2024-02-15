import copy
import sys

import torch
import torch.distributions as dist
import torchvision
from torch.utils.data import DataLoader

sys.path.append('../../')
from typing import List, Tuple

from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


def _p_update(vanilla_net: VanillaBnnLinear, p: list, eps: float) -> None:
    prior_loss = torch.tensor(0.0).requires_grad_(True).to(TORCH_DEVICE)
    for idx, param in enumerate(vanilla_net.parameters()):
        ll_grad = param.grad
        prior_loss = prior_loss + torch.neg(torch.mean(dist.Normal(0, 1).log_prob(param)))
        prior_grad = torch.autograd.grad(outputs=prior_loss, inputs=param)[0]
        potential_energy_update = ll_grad + prior_grad
        p[idx] = vanilla_net.update_param(p[idx], potential_energy_update, eps)

    vanilla_net.zero_grad()

def _q_update(vanilla_net: VanillaBnnLinear, p: list, eps: float) -> None:
    for idx, param in enumerate(vanilla_net.parameters()):
        new_val = vanilla_net.add_noise(param, p[idx], eps)
        with torch.no_grad():
            param.copy_(new_val)

def _get_batch(data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    data = next(iter(data_loader))
    batch_data, batch_target = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)

    return batch_data, batch_target


def _get_nll_loss(criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader, vanilla_net: VanillaBnnLinear) -> torch.Tensor:
    batch_data, batch_target = _get_batch(data_loader)
    # Forward pass
    y_hat = vanilla_net(batch_data)
    closs = criterion(y_hat, batch_target)
    vanilla_net.zero_grad()
    closs.backward()

    return closs

# this is wrong somehow
def _get_energy(vanilla_net: VanillaBnnLinear, q: list, p: list, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    # save the current parameters
    start_params = vanilla_net.get_params()

    # first update the nk params
    for idx, param in enumerate(vanilla_net.parameters()):
        with torch.no_grad():
            param.copy_(copy.deepcopy(q[idx]))

    # compute the potential energy
    batch_data, batch_target = _get_batch(data_loader)
    closs = criterion(vanilla_net(batch_data), batch_target)
    prior_loss = torch.tensor(0.0).to(TORCH_DEVICE)
    for idx, param in enumerate(vanilla_net.parameters()):
        prior_loss += torch.neg(torch.mean(dist.Normal(0, 1).log_prob(param)))
    potential_energy = closs + prior_loss

    # compute the kinetic energy
    kinetic_energy = torch.tensor(0.0).to(TORCH_DEVICE)
    for idx, p_val in enumerate(p):
        kinetic_energy = kinetic_energy + torch.sum(p_val * p_val) / 2

    # reset the parameters
    vanilla_net.set_params(start_params)
    return potential_energy + kinetic_energy

def train_mnist_vanilla(train_set: torchvision.datasets.mnist) -> VanillaBnnLinear:
    vanilla_net = VanillaBnnLinear()
    vanilla_net.to(TORCH_DEVICE)

    # hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    num_burnin_epochs = 5
    num_epochs = 10
    batch_size = 128
    print_freq = 500 // (batch_size / 64)
    momentum_std = 0.025
    # make L = len(train_data) / batch_size
    L = len(train_set) // batch_size
    print(f'L: {L}')

    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    posterior_samples = []

    # Initialize the parameters with a standard normal --
    # q -> current net params, current q -> start net params
    current_q = []
    for param in vanilla_net.named_parameters():
        if 'weight' in param[0]:
            init_vals = torch.normal(mean=0.0, std=0.1, size=tuple(param[1].shape)).to(TORCH_DEVICE)
            param[1].data = torch.nn.parameter.Parameter(init_vals)
    for param in vanilla_net.parameters():
        current_q.append(copy.deepcopy(param))

    running_loss_ce = 0.0
    for epoch in range(num_epochs):
        losses, p = [], []
        for param in vanilla_net.parameters():
            p.append(dist.Normal(0, momentum_std).sample(param.shape).to(TORCH_DEVICE))
        current_p = copy.deepcopy(p)

        # ------- half step for momentum -------
        closs = _get_nll_loss(criterion, data_loader, vanilla_net)
        _p_update(vanilla_net, p, lr / 2)
        for i in range(L):
            # ------------------- START q + eps * p -------------------
            _q_update(vanilla_net, p, lr)
            # ------------------- END q + eps * p -------------------

            # ------------------- START p - eps * grad_U(q) -------------------
            if i == L - 1:
                break
            closs = _get_nll_loss(criterion, data_loader, vanilla_net)
            _p_update(vanilla_net, p, lr)
            # ------------------- END p - eps * grad_U(q) -------------------

            losses.append(closs.item())
            running_loss_ce += closs.item()
            if i % print_freq == print_freq - 1:
                print(f'[epoch {epoch + 1}, batch {i + 1}] cross_entropy loss: {running_loss_ce / (batch_size * 500)}')
                running_loss_ce = 0.0

        # ------- final half step for momentum -------
        closs = _get_nll_loss(criterion, data_loader, vanilla_net)
        _p_update(vanilla_net, p, lr / 2)
        for idx, p_val in enumerate(p):
            p[idx] = -p_val

        # metropolis-hastings acceptance step
        q = vanilla_net.get_params()
        initial_energy = _get_energy(vanilla_net, current_q, current_p, criterion, data_loader)
        end_energy = _get_energy(vanilla_net, q, p, criterion, data_loader)
        acceptance_prob = min(1, torch.exp(end_energy - initial_energy))
        print(f'Acceptance probability: {acceptance_prob}')

        if torch.rand(1).to(TORCH_DEVICE) < acceptance_prob:
            current_q = q
            current_p = p
            if epoch > num_burnin_epochs - 1:
                print(f'Accepted sample at epoch {epoch + 1}...')
                posterior_samples.append(current_q)
        vanilla_net.set_params(current_q)
        vanilla_net.zero_grad()

    return vanilla_net, posterior_samples

def test_mnist_bnn(vanilla_net: VanillaBnnLinear, test_set: torchvision.datasets.mnist, posterior_samples: List[torch.tensor]) -> float:
    accuracies = []
    for sample in posterior_samples:
        vanilla_net.set_params(sample)
        vanilla_net.eval()
        batch_size = 32
        data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
        losses, correct, total = [], 0, 0

        for data, target in data_loader:
            batch_data_test, batch_target_test = data.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
            y_hat = vanilla_net(batch_data_test)
            loss = torch.nn.functional.cross_entropy(y_hat, batch_target_test)
            losses.append(loss.item())
            # also compute accuracy
            _, predicted = torch.max(y_hat, 1)
            total += batch_target_test.size(0)
            correct += (predicted == batch_target_test).sum().item()

        accuracies.append(100 * correct / total)
    print(f"Accuracies: {accuracies}")

    return sum(accuracies) / len(accuracies)


def test_mnist_vanilla(vanilla_net: VanillaBnnLinear, test_set: torchvision.datasets.mnist):
    vanilla_net.eval()
    batch_size = 32
    data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    losses, correct, total = [], 0, 0

    for data, target in data_loader:
        batch_data_test, batch_target_test = data.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
        y_hat = vanilla_net(batch_data_test)
        loss = torch.nn.functional.cross_entropy(y_hat, batch_target_test)
        losses.append(loss.item())
        # also compute accuracy
        _, predicted = torch.max(y_hat, 1)
        total += batch_target_test.size(0)
        correct += (predicted == batch_target_test).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    return 100 * correct / total


print(f"Using device: {TORCH_DEVICE}")
train_data, test_data = load_mnist()
vanilla_network, samples = train_mnist_vanilla(train_data)
mean_accuracy = test_mnist_bnn(vanilla_network, test_data, samples)
# accuracy = test_mnist_vanilla(vanilla_network, test_data)
print(f'Mean accuracy of the network on the 10000 test images: {mean_accuracy} %')