import sys
from typing import List, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader

sys.path.append('../../')
sys.path.append('../')

from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.HMC.bnn import VanillaBNN
from probabilistic.HMC.losses import (cross_entropy_likelihood,
                                      neg_log_normal_pdf)


class HyperparamsHMC:
    def __init__(self, num_epochs, num_burnin_epochs, lf_step, momentum_var = torch.tensor(1.0), prior_mu = torch.tensor(0.0), prior_var = torch.tensor(1.0),
                 ll_var = torch.tensor(1.0), steps_per_epoch = -1, batch_size = 1, batches_per_epoch = -1, gradient_norm_bound = -1, dp_sigma = 1.0):
        self.num_epochs = num_epochs
        self.num_burnin_epochs = num_burnin_epochs
        self.lf_step = lf_step
        self.momentum_var = momentum_var.to(TORCH_DEVICE)
        self.prior_mu = prior_mu.to(TORCH_DEVICE)
        self.prior_var = prior_var.to(TORCH_DEVICE)
        self.ll_var = ll_var.to(TORCH_DEVICE)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.gradient_norm_bound = gradient_norm_bound
        self.sigma = dp_sigma

class HamiltonianMonteCarlo:
    def __init__(self, hyperparams: HyperparamsHMC, net: VanillaBNN, loss_fn: callable, likelihood: callable, prior: callable) -> None:
        self.hps = hyperparams
        self.net = net
        self.loss_fn = loss_fn
        self.likelihood = likelihood
        self.prior = prior

    def train_hmc(self, train_data: torchvision.datasets.mnist):
        accepted_samples = []
        current_q = self.net.get_weights()
        train_set = DataLoader(train_data, batch_size=self.hps.batch_size, shuffle=True)
        eps = self.hps.lf_step
        self.net.train()

        # train loop
        for epoch in range(self.hps.num_epochs):
            # ------- INITIALIZATION -------
            q = current_q.detach().clone().requires_grad_(True)
            # resample p from a normal distribution with mean 0 and variance from config
            p = torch.normal(mean=0, std=torch.sqrt(self.hps.momentum_var), size=(current_q.shape[0],)).requires_grad_(True)
            p, q = p.to(TORCH_DEVICE), q.to(TORCH_DEVICE)
            current_p = p.detach().clone()

            print(f'Epoch {epoch}...')
            x_train, y_train = self._get_batch(train_set)
            # ------- END INITIALIZATION -------

            p = p - (eps / 2) * self.grad_U(q, x_train, y_train)
            for leapfrog_step in range(self.hps.steps_per_epoch):
                q = q + eps * p
                x_train, y_train = self._get_batch(train_set)
                self.net.set_weights(q)
                if leapfrog_step != self.hps.steps_per_epoch - 1:
                    p = p - eps * self.grad_U(q, x_train, y_train)
            p = p - (eps / 2) * self.grad_U(q, x_train, y_train)
            p = -p


            u = torch.rand(1).to(TORCH_DEVICE)
            train_set = DataLoader(train_data, batch_size=5000, shuffle=True)
            x_train, y_train = self._get_batch(train_set)
            energy_start = self.potential_energy(current_q, x_train, y_train) + self.kinetic_energy(current_p)
            energy_end = self.potential_energy(q, x_train, y_train) + self.kinetic_energy(p)

            acceptance_probability = min(1, torch.exp(energy_end - energy_start))
            print(f'Acceptance probability: {acceptance_probability}')

            if u < acceptance_probability:
                current_q = q.detach().clone()
                current_p = p.detach().clone()

                print(f'Accepted sample {current_q}')
                if epoch > self.hps.num_burnin_epochs:
                    accepted_samples.append(current_q)

            self.net.set_weights(current_q)

        return accepted_samples

    def test_hmc(self, param_samples: List[torch.Tensor], test_data: torchvision.datasets.mnist):
        self.net.eval()
        data_loader = DataLoader(test_data, batch_size=5000, shuffle=True)
        num_param_samples = len(param_samples)

        predictive_distribution = []
        criterion = torch.nn.Softmax(dim=1)
        for idx, param_sample in enumerate(param_samples):
            if idx % 7 == 0:
                print(f'Predicting with weight sample {idx}...')
            self.net.set_weights(param_sample)
            sample_predictions = []
            with torch.no_grad():
                for data, _ in data_loader:
                    batch_data_test = data.to(TORCH_DEVICE)
                    logits = self.net(batch_data_test)
                    batch_softmax = criterion(logits)
                    print(f"Shape of batch softmax: {batch_softmax.shape}")
                    print(f"Batch softmax: {batch_softmax[:5, :]}")
                    sample_predictions.append(batch_softmax)

            # flatten it into a single tensor of size test_size x labels
            predictive_distribution.append(torch.cat(sample_predictions))

        print(f"shape of all predictions: {num_param_samples} x {predictive_distribution[0].shape}")
        predicted_label_mean = torch.zeros_like(predictive_distribution[0])
        for i in range(num_param_samples):
            predicted_label_mean += predictive_distribution[i]
        predicted_label_mean /= num_param_samples

        # Get the most often encountered class for each example
        # predicted_label_mean = predictive_distribution.mean(dim=2)
        print("shape of mean predictions: ", predicted_label_mean.shape)
        predicted_labels = torch.argmax(predicted_label_mean, dim=1)
        print("shape of predicted labels: ", predicted_labels.shape)
        # compute the accuracy
        accuracy = (predicted_labels == test_data.targets.to(TORCH_DEVICE)).sum().item() / len(test_data.targets)

        return accuracy


    def potential_energy(self, q: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        prior_val = self.prior(q, self.hps.prior_mu, torch.sqrt(self.hps.prior_var))
        likelihood_val = self.likelihood(x, y, self.net, self.loss_fn)

        return likelihood_val + prior_val

    def grad_U(self, q: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        prior_val = self.prior(q, self.hps.prior_mu, torch.sqrt(self.hps.prior_var))
        likelihood_val = self.likelihood(x, y, self.net, self.loss_fn)
        print(f"Cross entropy: {likelihood_val}")

        grad_prior = torch.autograd.grad(outputs=prior_val, inputs=q)[0]
        likelihood_val.backward()
        grad_likelihood = self.net.get_weight_grads()

        # reset the gradients
        self.net.set_zero_grads()

        return grad_likelihood + grad_prior

    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        return (p ** 2 / (2 * self.hps.momentum_var)).sum()

    def _get_batch(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = next(iter(data_loader))
        return batch[0].to(TORCH_DEVICE), batch[1].to(TORCH_DEVICE)


def run_experiment():
    hps = HyperparamsHMC(num_epochs=4,
                         num_burnin_epochs=1,
                         lf_step=0.002,
                         momentum_var=torch.tensor(1.0),
                         prior_mu=torch.tensor(0.0),
                         prior_var=torch.tensor(1.0),
                         steps_per_epoch=20,
                         batch_size=1000)

    likelihood = cross_entropy_likelihood
    prior = neg_log_normal_pdf
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    net = VanillaBNN().to(TORCH_DEVICE)

    hmc = HamiltonianMonteCarlo(hps, net, loss_fn, likelihood, prior)
    train_data, test_data = load_mnist("../../")

    hmc_samples = hmc.train_hmc(train_data)

    accuracy = hmc.test_hmc(hmc_samples, test_data)
    print(f'Accuracy: {accuracy}')

run_experiment()
