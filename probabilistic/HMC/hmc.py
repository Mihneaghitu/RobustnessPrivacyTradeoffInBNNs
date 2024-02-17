import copy
import sys
from typing import List, Tuple

import torch
import torch.distributions as dist
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

sys.path.append('../../')

from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.HMC.vanilla_bnn import HyperparamsHMC, VanillaBnnLinear


class HamiltonianMonteCarlo:
    def __init__(self, net: VanillaBnnLinear, hyperparameters: HyperparamsHMC) -> None:
        self.net = net
        self.hps = hyperparameters


    def train_mnist_vanilla(self, train_set: torchvision.datasets.mnist) -> List[torch.tensor]:
        # Just don't ask
        print_freq = 500 // (self.hps.batch_size / 64)

        data_loader = DataLoader(train_set, batch_size=self.hps.batch_size, shuffle=True)
        posterior_samples = []

        # Initialize the parameters with a standard normal --
        # q -> current net params, current q -> start net params
        current_q = []
        for param in self.net.named_parameters():
            if 'weight' in param[0]:
                init_vals = torch.normal(mean=0.0, std=0.1, size=tuple(param[1].shape)).to(TORCH_DEVICE)
                param[1].data = torch.nn.parameter.Parameter(init_vals)
        for param in self.net.parameters():
            current_q.append(copy.deepcopy(param))

        running_loss_ce = 0.0
        for epoch in range(self.hps.num_epochs):
            losses, p = [], []
            for param in self.net.parameters():
                p.append(dist.Normal(0, self.hps.momentum_std).sample(param.shape).to(TORCH_DEVICE))
            current_p = copy.deepcopy(p)

            # ------- half step for momentum -------
            closs = self._get_nll_loss(self.hps.criterion, data_loader)
            self._p_update(p, self.hps.step_size / 2)
            for i in range(self.hps.lf_steps):
                # ------------------- START q + eps * p -------------------
                self._q_update(p, self.hps.step_size)
                # ------------------- END q + eps * p -------------------

                # ------------------- START p - eps * grad_U(q) -------------------
                if i == self.hps.lf_steps - 1:
                    break
                closs = self._get_nll_loss(self.hps.criterion, data_loader)
                self._p_update(p, self.hps.step_size)
                # ------------------- END p - eps * grad_U(q) -------------------

                losses.append(closs.item())
                running_loss_ce += closs.item()
                if i % print_freq == print_freq - 1:
                    print(f'[epoch {epoch + 1}, batch {i + 1}] cross_entropy loss: {running_loss_ce / (self.hps.batch_size * print_freq)}')
                    running_loss_ce = 0.0

            # ------- final half step for momentum -------
            closs = self._get_nll_loss(self.hps.criterion, data_loader)
            self._p_update(p, self.hps.step_size / 2)
            for idx, p_val in enumerate(p):
                p[idx] = -p_val

            # metropolis-hastings acceptance step
            q = self.net.get_params()
            initial_energy = self._get_energy(current_q, current_p, self.hps.criterion, data_loader)
            end_energy = self._get_energy(q, p, self.hps.criterion, data_loader)
            acceptance_prob = min(1, torch.exp(end_energy - initial_energy))
            print(f'Acceptance probability: {acceptance_prob}')

            if torch.rand(1).to(TORCH_DEVICE) < acceptance_prob:
                current_q = q
                current_p = p
                if epoch > self.hps.num_burnin_epochs - 1:
                    print(f'Accepted sample at epoch {epoch + 1}...')
                    posterior_samples.append(current_q)
            self.net.set_params(current_q)
            self.net.zero_grad()

        return posterior_samples

    def test_mnist_bnn(self, test_set: torchvision.datasets.mnist, posterior_samples: List[torch.tensor]) -> float:
        accuracies = []
        for sample in posterior_samples:
            self.net.set_params(sample)
            self.net.eval()
            batch_size = 32
            data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
            losses, correct, total = [], 0, 0

            for data, target in data_loader:
                batch_data_test, batch_target_test = data.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
                y_hat = self.net(batch_data_test)
                loss = F.cross_entropy(y_hat, batch_target_test)
                losses.append(loss.item())
                # also compute accuracy -- torch.max returns (values, indices)
                _, predicted = torch.max(y_hat, 1)
                total += batch_target_test.size(0)
                correct += (predicted == batch_target_test).sum().item()

            accuracies.append(100 * correct / total)
        print(f"Accuracies: {accuracies}")

        return sum(accuracies) / len(accuracies)


    def test_hmc_with_average_logits(self, test_set: torchvision.datasets.mnist, posterior_samples: List[torch.tensor]) -> float:
        average_logits = torch.zeros(len(test_set), 10).to(TORCH_DEVICE)
        for sample in posterior_samples:
            self.net.set_params(sample)
            self.net.eval()
            batch_size = 32
            data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

            sample_results = torch.tensor([]).to(TORCH_DEVICE)
            for data, _ in data_loader:
                batch_data_test  = data.to(TORCH_DEVICE)
                y_hat = self.net(batch_data_test)
                sample_results = torch.cat((sample_results, y_hat), dim=0)
            average_logits += sample_results / len(posterior_samples)

        print(f"Len posterior samples: {len(posterior_samples)}")
        correct, total = 0, test_set.targets.size(0)
        for i in range(test_set.targets.size(0)):
            avg_logit = average_logits[i]
            index_of_max_logit = torch.argmax(avg_logit)
            if index_of_max_logit == test_set.targets[i]:
                correct += 1

        return 100 * correct / total

    # ---------------------------------------------------------
    # --------------------- Helper functions ------------------
    # ---------------------------------------------------------

    def _p_update(self, p: list, eps: float) -> None:
        prior_loss = torch.tensor(0.0).requires_grad_(True).to(TORCH_DEVICE)
        for idx, param in enumerate(self.net.parameters()):
            ll_grad = param.grad
            prior_loss = prior_loss + torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
            prior_grad = torch.autograd.grad(outputs=prior_loss, inputs=param)[0]
            potential_energy_update = ll_grad + prior_grad
            p[idx] = self.net.update_param(p[idx], potential_energy_update, eps)

        self.net.zero_grad()

    def _q_update(self, p: list, eps: float) -> None:
        for idx, param in enumerate(self.net.parameters()):
            new_val = self.net.add_noise(param, p[idx], eps)
            with torch.no_grad():
                param.copy_(new_val)

    def _get_batch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        data = next(iter(data_loader))
        batch_data, batch_target = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)

        return batch_data, batch_target


    def _get_nll_loss(self, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        batch_data, batch_target = self._get_batch(data_loader)
        # Forward pass
        y_hat = self.net(batch_data)
        closs = criterion(y_hat, batch_target)
        self.net.zero_grad()
        closs.backward()

        return closs

    # this is wrong somehow
    def _get_energy(self, q: list, p: list, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        # save the current parameters
        start_params = self.net.get_params()

        # first update the nk params
        for idx, param in enumerate(self.net.parameters()):
            with torch.no_grad():
                param.copy_(copy.deepcopy(q[idx]))

        # compute the potential energy
        batch_data, batch_target = self._get_batch(data_loader)
        closs = criterion(self.net(batch_data), batch_target)
        prior_loss = torch.tensor(0.0).to(TORCH_DEVICE)
        for idx, param in enumerate(self.net.parameters()):
            prior_loss += torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
        potential_energy = closs + prior_loss

        # compute the kinetic energy
        kinetic_energy = torch.tensor(0.0).to(TORCH_DEVICE)
        for idx, p_val in enumerate(p):
            kinetic_energy = kinetic_energy + torch.sum(p_val * p_val) / 2

        # reset the parameters
        self.net.set_params(start_params)
        return potential_energy + kinetic_energy


# Setup
print(f"Using device: {TORCH_DEVICE}")
# NOTE: lf_steps = len(train_data) // batch_size = 60000 // batch_size = 468 -- this is to see all the dataset for one numerical integration step
hyperparams = HyperparamsHMC(num_epochs=10, num_burnin_epochs=5, step_size=0.001, lf_steps=468, criterion=torch.nn.CrossEntropyLoss(),
                             batch_size=128, momentum_std=0.025)
VANILLA_BNN = VanillaBnnLinear().to(TORCH_DEVICE)
hmc = HamiltonianMonteCarlo(VANILLA_BNN, hyperparams)

# Train and test
train_data, test_data = load_mnist("../../")
samples = hmc.train_mnist_vanilla(train_data)
mean_accuracy = hmc.test_mnist_bnn(test_data, samples)
print(f'Mean accuracy of the network on the 10000 test images: {mean_accuracy} %')
acc_with_average_logits = hmc.test_hmc_with_average_logits(test_data, samples)
print(f'Accuracy with average logits: {acc_with_average_logits} %')
