import copy
import sys
from typing import List, Tuple

import torch
import torch.distributions as dist
import torch.nn.functional as F
import torchvision
import wandb
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader

sys.path.append('../../')

from globals import LOGGER_TYPE, TORCH_DEVICE, LoggerType
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


class HamiltonianMonteCarlo:
    def __init__(self, net: VanillaBnnLinear, hyperparameters: HyperparamsHMC) -> None:
        self.net = net
        self.hps = hyperparameters

    def train_bnn(self, train_set: torchvision.datasets.mnist) -> List[torch.tensor]:
        print_freq = self.hps.lf_steps - 1

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
            self.__run_schedule(epoch + 1) # decay the step size
            for param in self.net.parameters():
                p.append(dist.Normal(0, self.hps.momentum_std).sample(param.shape).to(TORCH_DEVICE))
            current_p = copy.deepcopy(p)

            # ------- half step for momentum -------
            closs = self.__get_nll_loss(self.hps.criterion, data_loader)
            self.__p_update(p, self.hps.step_size / 2)
            for i in range(self.hps.lf_steps):
                # ------------------- START q + eps * p -------------------
                self.__q_update(p, self.hps.step_size)
                # ------------------- END q + eps * p -------------------

                # ------------------- START p - eps * grad_U(q) -------------------
                if i == self.hps.lf_steps - 1:
                    break
                closs = self.__get_nll_loss(self.hps.criterion, data_loader)
                self.__p_update(p, self.hps.step_size)
                # ------------------- END p - eps * grad_U(q) -------------------

                losses.append(closs.item())
                running_loss_ce += closs.item()
                if i % print_freq == print_freq - 1:
                    print(f'[epoch {epoch + 1}, batch {i + 1}] cross_entropy loss: {running_loss_ce / (self.hps.batch_size * print_freq)}')
                    running_loss_ce = 0.0
            if LOGGER_TYPE == LoggerType.WANDB:
                wandb.log({'cross_entropy_loss': losses[-1]})
                wandb.log({'epoch': epoch + 1})

            # ------- final half step for momentum -------
            closs = self.__get_nll_loss(self.hps.criterion, data_loader)
            self.__p_update(p, self.hps.step_size / 2)
            for idx, p_val in enumerate(p):
                p[idx] = -p_val

            # metropolis-hastings acceptance step
            q = self.net.get_params()
            if not self.hps.run_dp:
                initial_energy = self.__get_energy(current_q, current_p, data_loader)
                end_energy = self.__get_energy(q, p, data_loader)
                acceptance_prob = min(1, torch.exp(end_energy - initial_energy))
            else:
                acceptance_prob = min(1, self.__get_dp_energy(q, p, current_q, current_p, data_loader))

            print(f'Acceptance probability: {acceptance_prob}')
            if LOGGER_TYPE == LoggerType.WANDB:
                wandb.log({'acceptance_probability': acceptance_prob})

            if epoch <= self.hps.num_burnin_epochs - 1 or dist.Uniform(0, 1).sample().to(TORCH_DEVICE) < acceptance_prob:
                current_q = q
                current_p = p
                if epoch > self.hps.num_burnin_epochs - 1:
                    posterior_samples.append(current_q)
            self.net.set_params(current_q)
            self.net.zero_grad()

        return posterior_samples

    def test_hmc_with_average_logits(self, test_set: torchvision.datasets.mnist, posterior_samples: List[torch.tensor]) -> float:
        average_logits = torch.zeros(len(test_set), 10).to(TORCH_DEVICE)
        for sample in posterior_samples:
            self.net.set_params(sample)
            self.net.eval()
            batch_size = 32
            data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

            sample_results = torch.tensor([]).to(TORCH_DEVICE)
            for data, _ in data_loader:
                batch_data_test  = data.to(TORCH_DEVICE)
                y_hat = self.net(batch_data_test)
                sample_results = torch.cat((sample_results, y_hat), dim=0)
            average_logits += sample_results / len(posterior_samples)

        correct, total = 0, test_set.targets.size(0)
        for i in range(test_set.targets.size(0)):
            avg_logit = average_logits[i]
            index_of_max_logit = torch.argmax(avg_logit)
            if index_of_max_logit == test_set.targets[i]:
                correct += 1

        print(f"Accuracy with average logits: {100 * correct / total} %")
        if LOGGER_TYPE == LoggerType.WANDB:
            wandb.log({'accuracy_with_average_logits': 100 * correct / total})

        return 100 * correct / total

    def get_delta_dp_bound(self, eps: float) -> float:
        mu = (self.hps.num_epochs / (2 * self.hps.tau_l)) + (self.hps.num_epochs * (self.hps.lf_steps + 1) / (2 * self.hps.tau_g))

        delta = 0.5 * (torch.erfc((eps - mu) / (2 * torch.sqrt(mu))) - torch.exp(eps) * torch.erfc((eps + mu) / (2 * torch.sqrt(mu))))

        return float(delta.item())

    # ---------------------------------------------------------
    # -------------------- Helper functions -------------------
    # ---------------------------------------------------------

    def __p_update(self, p: list, eps: float) -> None:
        prior_loss = torch.tensor(0.0).requires_grad_(True).to(TORCH_DEVICE)
        for idx, param in enumerate(self.net.parameters()):
            ll_grad = param.grad
            prior_loss = prior_loss + torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
            prior_grad = torch.autograd.grad(outputs=prior_loss, inputs=param)[0]
            potential_energy_grad = ll_grad + prior_grad
            if self.hps.run_dp:
                total_batch_noise = dist.Normal(0, 2 * self.hps.tau_g * self.hps.grad_clip_bound).sample(param.shape).to(TORCH_DEVICE)
                potential_energy_grad += total_batch_noise / self.hps.batch_size
            p[idx] = self.net.update_param(p[idx], potential_energy_grad, eps)

        self.net.zero_grad()

    def __q_update(self, p: list, eps: float) -> None:
        for idx, param in enumerate(self.net.parameters()):
            new_val = self.net.add_noise(param, p[idx], eps)
            with torch.no_grad():
                param.copy_(new_val)

    def __get_nll_loss(self, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        batch_data, batch_target = self.__get_batch(data_loader)
        # Forward pass
        y_hat = self.net(batch_data)
        closs = criterion(y_hat, batch_target)
        self.net.zero_grad()
        if self.hps.run_dp:
            params = {k: v.detach() for k, v in self.net.named_parameters()}
            buffers = {k: v.detach() for k, v in self.net.named_buffers()}
            compute_grad = grad(self.__compute_per_sample_loss)
            compute_per_sample_grads = vmap(compute_grad, in_dims=(None, None, 0, 0))
            per_sample_grads = compute_per_sample_grads(params, buffers, batch_data, batch_target)

            per_layer_mean_clipped_grads = []
            for batch_grads in per_sample_grads.values():
                clip_per_sample_grad = vmap(self.__clip, in_dims=(0, None), out_dims=0)
                clipped_per_sample_grads = clip_per_sample_grad(batch_grads, self.hps.grad_clip_bound)
                per_layer_mean_clipped_grads.append(torch.mean(clipped_per_sample_grads, dim=0)) # average over samples
            for idx, param in enumerate(self.net.parameters()):
                param.grad = per_layer_mean_clipped_grads[idx]
            # Now we are exactly in the same situation as in the non-DP case, i.e. with the mean batch gradients inside the
            # parameters' grad attributes, with the exception that everything was clipped to ensure privacy
        else:
            closs.backward()

        return closs

    def __get_energy(self, q: list, p: list, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        # save the current parameters
        start_params = self.net.get_params()

        # first update the nk params
        self.net.set_params(q)

        # compute the potential energy
        batch_data, batch_target = self.__get_batch(data_loader)
        closs = self.hps.criterion(self.net(batch_data), batch_target)
        prior_loss = torch.tensor(0.0).to(TORCH_DEVICE)
        for _, param in enumerate(self.net.parameters()):
            prior_loss += torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
        potential_energy = closs + prior_loss

        # compute the kinetic energy
        kinetic_energy = torch.tensor(0.0).to(TORCH_DEVICE)
        for _, p_val in enumerate(p):
            kinetic_energy += torch.neg(torch.sum(torch.pow(p_val , 2)) / 2)

        # reset the parameters
        self.net.set_params(start_params)
        return potential_energy + kinetic_energy

    def __get_dp_energy(self, prop_q: list, prop_p: list, curr_q: list, curr_p: list, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        compute_per_sample_loss = vmap(self.hps.criterion, in_dims=(0, 0), out_dims=0)
        # save the current parameters
        start_params = self.net.get_params()

        batch_data, batch_target = self.__get_batch(data_loader)
        self.net.set_params(curr_q)
        initial_per_sample_nll = compute_per_sample_loss(self.net(batch_data), batch_target)
        self.net.set_params(prop_q)
        proposal_per_sample_nll = compute_per_sample_loss(self.net(batch_data), batch_target)

        # get the norm of the difference between the current and the proposal parameters, needed for the clip bound
        diff = torch.tensor([]).to(TORCH_DEVICE)
        for c_q, p_q in zip(curr_q, prop_q):
            diff = torch.cat((diff, torch.flatten(p_q - c_q)))

        #* compute the clipping bound, which is ||theta_prop - theta_curr||_2 * b_l, then use a vmap to clip the ratio for every sample
        clipping_bound_ll_ratio = self.hps.acceptance_clip_bound * torch.norm(diff)
        per_sample_clipped_ll_ratio = vmap(lambda nom, denom: self.__clip(nom / denom, clipping_bound_ll_ratio), in_dims=(0, 0), out_dims=0)
        clipped_ll_ratio_batch = torch.mean(per_sample_clipped_ll_ratio(proposal_per_sample_nll, initial_per_sample_nll))

        # compute the kinetic energy
        kinetic_energy_init = torch.tensor(0.0).to(TORCH_DEVICE)
        kinetic_energy_prop = torch.tensor(0.0).to(TORCH_DEVICE)
        for c_p, p_p in zip(curr_p, prop_p):
            kinetic_energy_init += torch.neg(torch.sum(torch.pow(c_p, 2)) / 2)
            kinetic_energy_prop += torch.neg(torch.sum(torch.pow(p_p, 2)) / 2)
        delta_p = kinetic_energy_prop - kinetic_energy_init

        # compute the prior term, omit the minus sign because when computing the ratio, it will cancel out
        init_prior_prob, proposal_prior_prob = torch.tensor(0.0).to(TORCH_DEVICE), torch.tensor(0.0).to(TORCH_DEVICE)
        for c_q, p_q in zip(curr_q, prop_q):
            init_prior_prob += torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(c_q))
            proposal_prior_prob += torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(p_q))
        prior_ratio_batch = proposal_prior_prob - init_prior_prob

        # noise term
        dp_sigma_acceptance = 2 * self.hps.tau_l * clipping_bound_ll_ratio
        psi = dist.Normal(0, dp_sigma_acceptance).sample().to(TORCH_DEVICE)

        # correction term
        correction_term = torch.pow(dp_sigma_acceptance, 2) / 2

        energy_delta = clipped_ll_ratio_batch + delta_p + prior_ratio_batch + psi
        energy_delta_corrected = energy_delta - correction_term

        # reset the parameters
        self.net.set_params(start_params)
        return torch.exp(energy_delta_corrected) # not to have the acceptance probability in log space

    def __clip(self, x: torch.Tensor, bound: float) -> torch.Tensor:
        return x * torch.min(torch.tensor(1), bound / torch.norm(x))

    def __compute_per_sample_loss(self, params: torch.Tensor, buffers: torch.Tensor, sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(self.net, (params, buffers), (batch,))
        loss = self.hps.criterion(predictions, targets)
        return loss

    def __get_batch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        data = next(iter(data_loader))
        batch_data, batch_target = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)

        return batch_data, batch_target

    def __run_schedule(self, curr_epoch: int) -> None:
        decay_step = self.hps.step_size * (1 - self.hps.lr_decay_magnitude) / (self.hps.num_epochs - self.hps.decay_epoch_start)
        if curr_epoch >= self.hps.decay_epoch_start:
            self.hps.step_size -= decay_step
