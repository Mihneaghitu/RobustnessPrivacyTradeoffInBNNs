import copy
import os
import sys
from typing import List, Tuple

import torch
import torch.distributions as dist
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')

from torch.func import functional_call, grad, vmap

from common.attack_types import AttackType
from globals import LOGGER_TYPE, TORCH_DEVICE, LoggerType
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import IbpAdversarialLoss, VanillaBnnLinear


class AdvHamiltonianMonteCarlo:
    def __init__(self, net: VanillaBnnLinear, hyperparameters: HyperparamsHMC, attack_type: AttackType = AttackType.FGSM):
        self.net = net
        self.hps = hyperparameters
        self.init_hps = copy.deepcopy(hyperparameters)
        self.attack_type = attack_type
        self.adv_generator, self.adv_criterion = None, None
        match attack_type:
            case AttackType.FGSM:
                self.adv_generator = self.__gen_fgsm_adv_examples
                self.adv_criterion = self.hps.criterion
            case AttackType.PGD:
                self.adv_generator = self.__gen_pgd_adv_examples
                self.adv_criterion = self.hps.criterion
            case AttackType.IBP:
                self.adv_generator = self.__gen_ibp_adv_examples
                self.adv_criterion = IbpAdversarialLoss(net, self.hps.criterion, self.hps.eps)
                assert self.hps.eps_warmup_epochs <= self.hps.num_burnin_epochs, "IBP requires that eps_warmup_epochs <= num_burnin_epochs"
                assert self.hps.alpha_warmup_epochs <= self.hps.num_burnin_epochs, "IBP requires that alpha_warmup_epochs <= num_burnin_epochs"
        self.hps.eps, self.hps.alpha = 0, 1

    def train_with_restarts(self, train_set: Dataset, first_chain_from_trained: bool = False) -> List[torch.tensor]:
        posterior_samples_all_restarts = []
        if first_chain_from_trained: # first chain from a trained network
            # not needed when initializing from a trained network
            self.hps.num_burnin_epochs, self.hps.alpha_warmup_epochs, self.hps.eps_warmup_epochs = 0, 0, 0
            self.hps.eps, self.hps.alpha = self.init_hps.eps, self.init_hps.alpha_pre_trained
            self.hps.step_size = self.init_hps.step_size_pre_trained
            posterior_samples_all_restarts += self.train_bnn(train_set, from_trained=first_chain_from_trained)
            self.hps.num_chains -= 1

        for _ in range(self.hps.num_chains):
            self.hps = copy.deepcopy(self.init_hps)
            self.hps.eps, self.hps.alpha = 0, 1
            posterior_samples_all_restarts += self.train_bnn(train_set)

        return posterior_samples_all_restarts

    def train_bnn(self, train_set: Dataset, from_trained: bool = False) -> List[torch.tensor]:
        print_freq = self.hps.lf_steps - 1
        data_loader = DataLoader(train_set, batch_size=self.hps.batch_size, shuffle=True)
        posterior_samples = []

        # q -> current net params, current q -> start net params
        root_dir = __file__.rsplit('/', 3)[0]
        init_file = (os.path.abspath(root_dir + "pre_trained/vanilla_network_ibp_dp.pt") if self.hps.run_dp
                     else os.path.abspath(root_dir + "pre_trained/vanilla_network_ibp.pt"))
        current_q = self.__init_params(from_trained=from_trained, path=init_file)

        running_loss_ce, running_loss_ce_adv = 0.0, 0.0
        for epoch in range(self.hps.num_epochs):
            losses, p = [], []
            for param in self.net.parameters():
                p.append(dist.Normal(0, self.hps.momentum_std).sample(param.shape).to(TORCH_DEVICE))
            current_p = copy.deepcopy(p)

            # ------- half step for momentum -------
            self.__run_schedule(epoch)
            closs, closs_adv = self.__p_update(data_loader, p, self.hps.step_size / 2)
            running_loss_ce += closs.item()
            running_loss_ce_adv += closs_adv.item()
            for i in range(self.hps.lf_steps):
                self.__run_schedule(epoch)
                # ------------------- UPDATE q_new = q + lf_ss * p -------------------
                self.__q_update(p, self.hps.step_size)

                # ------------------- UPDATE p_new = p - lf_ss * (aplha * grad_U(q) + (1 - alpha) grad_U_adv(q)) -------------------
                if i == self.hps.lf_steps - 1:
                    break
                closs, closs_adv = self.__p_update(data_loader, p, self.hps.step_size)

                # -------------------- Logging --------------------
                losses.append((closs.item(), closs_adv.item()))
                running_loss_ce += closs.item()
                running_loss_ce_adv += closs_adv.item()
                if i % print_freq == print_freq - 1:
                    print(f'[epoch {epoch + 1}, leapfrog step {i + 1}] cross_entropy loss: {running_loss_ce / (self.hps.batch_size * print_freq)}')
                    print(f'[epoch {epoch + 1}, leapfrog step {i + 1}] ce_adversarial loss: {running_loss_ce_adv / (self.hps.batch_size * print_freq)}')
                    running_loss_ce, running_loss_ce_adv = 0.0, 0.0

            if LOGGER_TYPE == LoggerType.WANDB:
                wandb.log({'cross_entropy_loss': losses[-1][0]})
                wandb.log({'epoch': epoch + 1})
                wandb.log({'cross_entropy_adversarial_loss': losses[-1][1]})

            # -------------- Final half step for momentum --------------
            closs, closs_adv = self.__p_update(data_loader, p, self.hps.step_size / 2)
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

    def test_hmc_with_average_logits(self, test_set: Dataset, posterior_samples: List[torch.tensor]) -> float:
        mean_logits = torch.zeros(len(test_set), self.net.get_num_classes()).to(TORCH_DEVICE)
        for sample in posterior_samples:
            self.net.set_params(sample)
            self.net.eval()
            batch_size = 32
            data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

            sample_results = torch.tensor([]).to(TORCH_DEVICE)
            for data, _ in data_loader:
                batch_data_test  = data.to(TORCH_DEVICE)
                index_of_max_logit = self.net(batch_data_test)
                sample_results = torch.cat((sample_results, index_of_max_logit), dim=0)
            mean_logits += sample_results / len(posterior_samples)

        correct, total = 0, test_set.targets.size(0)
        if self.net.get_num_classes() > 2:
            for i in range(test_set.targets.size(0)):
                avg_logit = mean_logits[i]
                index_of_max_logit = torch.argmax(avg_logit)
                if index_of_max_logit == test_set.targets[i].to(TORCH_DEVICE):
                    correct += 1
        # Binary classification
        else:
            sigmoid = torch.nn.Sigmoid()
            for i in range(test_set.targets.size(0)):
                pred = sigmoid(mean_logits[i])
                if (pred > 0.5 and test_set.targets[i] > 0.5) or (pred < 0.5 and test_set.targets[i] < 0.5):
                    correct += 1

        return 100 * correct / total

    def get_delta_dp_bound(self, eps: float) -> float:
        mu = (self.hps.num_epochs / (2 * self.hps.tau_l)) + (self.hps.num_epochs * (self.hps.lf_steps + 1) / (2 * self.hps.tau_g))

        delta = 0.5 * (torch.erfc((eps - mu) / (2 * torch.sqrt(mu))) - torch.exp(eps) * torch.erfc((eps + mu) / (2 * torch.sqrt(mu))))

        return float(delta.item())

    # ---------------------------------------------------------
    # -------------------- Helper functions -------------------
    # ---------------------------------------------------------

    def __p_update(self, data_loader: torch.utils.data.DataLoader, p: list, lf_step: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_data, batch_target = self.__get_batch(data_loader)
        closs = self.__get_nll_loss(self.hps.criterion, (batch_data, batch_target), adv=False)
        self.__advance_momentum(p, self.hps.alpha * lf_step, adv=False)
        closs_adv = self.__get_nll_loss(self.adv_criterion, (batch_data, batch_target), adv=True)
        self.__advance_momentum(p, (1 - self.hps.alpha) * lf_step, adv=True)
        # Add noise if DP
        if self.hps.run_dp:
            for idx, param in enumerate(self.net.parameters()):
                total_batch_noise = dist.Normal(0, 2 * self.hps.tau_g * self.hps.grad_clip_bound).sample(param.shape).to(TORCH_DEVICE)
                total_batch_noise /= self.hps.batch_size
                p[idx] = self.net.update_param(p[idx], total_batch_noise, lf_step)

        return closs, closs_adv

    def __advance_momentum(self, p: list, lr: float, adv=False) -> None:
        prior_loss = torch.tensor(0.0).requires_grad_(True).to(TORCH_DEVICE)
        prior_grad = torch.tensor(0.0).requires_grad_(True).to(TORCH_DEVICE)
        for idx, param in enumerate(self.net.parameters()):
            ll_grad = param.grad
            if not adv:
                prior_loss = prior_loss + torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
                prior_grad = torch.autograd.grad(outputs=prior_loss, inputs=param)[0]
                prior_grad /= self.hps.alpha
            potential_energy_grad = self.__get_dp_grads(ll_grad) + prior_grad
            p[idx] = self.net.update_param(p[idx], potential_energy_grad, lr)

        self.net.zero_grad()

    def __q_update(self, p: list, eps: float) -> None:
        for idx, param in enumerate(self.net.parameters()):
            new_val = self.net.add_noise(param, p[idx], eps)
            with torch.no_grad():
                param.copy_(new_val)

    def __get_nll_loss(self, criterion: torch.nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], adv = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_data, batch_target = batch
        if adv: # adversarial training
            batch_data = self.adv_generator(batch_data, batch_target)

        #! In the case of IBP, the adversarial examples generation is hidden and actually happens inside the loss function,
        #! when the ibp_forward() func is called to propagate the bounds.
        if adv and self.attack_type == AttackType.IBP:
            closs = criterion(batch_data, batch_target)
        else:
            y_hat = self.net(batch_data)
            closs = criterion(y_hat, batch_target)

        self.net.zero_grad()
        if self.hps.run_dp:
            params, buffers = None, None
            compute_grad = None
            if adv:
                params = {k: v.detach() for k, v in self.adv_criterion.named_parameters()}
                buffers = {k: v.detach() for k, v in self.adv_criterion.named_buffers()}
                compute_grad = grad(self.__compute_per_sample_loss_adv)
            else:
                params = {k: v.detach() for k, v in self.net.named_parameters()}
                buffers = {k: v.detach() for k, v in self.net.named_buffers()}
                compute_grad = grad(self.__compute_per_sample_loss_std)
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

    def __gen_pgd_adv_examples(self, batch_input: torch.Tensor, batch_target: torch.Tensor, iters=10):
        curr_input = copy.deepcopy(batch_input)
        for _ in range(iters) :
            curr_input.requires_grad = True
            y_hat = self.net(curr_input)

            self.net.zero_grad()
            loss = self.hps.criterion(y_hat, batch_target)
            loss.backward()

            cur_it_projected_adv_examples = copy.deepcopy(curr_input) + self.hps.eps * torch.sign(curr_input.grad.data)
            delta = torch.clamp(cur_it_projected_adv_examples - batch_input, min=-self.hps.eps, max=self.hps.eps)
            cur_it_projected_adv_examples = torch.clamp(batch_input + delta, min=0, max=1).detach_()
            curr_input = copy.deepcopy(cur_it_projected_adv_examples)

        return cur_it_projected_adv_examples

    def __gen_fgsm_adv_examples(self, batch_input: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        batch_input.requires_grad = True
        y_hat = self.net(batch_input)
        loss = self.hps.criterion(y_hat, batch_target)
        loss.backward()
        input_grads = copy.deepcopy(batch_input.grad.data)
        adv_examples = copy.deepcopy(batch_input)
        adv_examples = adv_examples + self.hps.eps * input_grads.sign()
        clamped_adv_examples = torch.clamp(adv_examples, 0, 1)
        batch_input.grad.zero_()
        batch_input.requires_grad = False
        self.net.zero_grad()

        return clamped_adv_examples

    def __gen_ibp_adv_examples(self, batch_input: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        #* This looks like a dummy function, and it is. Nevertheless, it's necessary for the sake of consistency
        return batch_input

    def __get_energy(self, q: list, p: list, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        # save the current parameters
        start_params = self.net.get_params()

        # first update the nk params
        self.net.set_params(q)

        # compute the potential energy
        batch_data, batch_target = self.__get_batch(data_loader)
        closs = self.hps.criterion(self.net(batch_data), batch_target)
        adv_ex = self.adv_generator(batch_data, batch_target)
        x = adv_ex if self.attack_type == AttackType.IBP else self.net(adv_ex)
        closs_adv = self.adv_criterion(x, batch_target)
        prior_loss = torch.tensor(0.0).to(TORCH_DEVICE)
        for _, param in enumerate(self.net.parameters()):
            prior_loss += torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
        potential_energy = self.hps.alpha * closs + (1 - self.hps.alpha) * closs_adv + prior_loss

        # compute the kinetic energy
        kinetic_energy = torch.tensor(0.0).to(TORCH_DEVICE)
        for _, p_val in enumerate(p):
            kinetic_energy += torch.neg(torch.sum(torch.pow(p_val, 2)) / 2)

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

    def __get_dp_grads(self, ll_grad: torch.Tensor) -> torch.Tensor:
        dp_grad = copy.deepcopy(ll_grad)
        if self.hps.run_dp:
            # clip the gradient norm (first term) and add noise (second term)
            dp_grad = self.__clip(dp_grad, self.hps.grad_clip_bound)
            dp_noise = dist.Normal(0, 2 * self.hps.tau_g * self.hps.grad_clip_bound).sample(dp_grad.shape).to(TORCH_DEVICE)
            # add the mean noise across the batch to grad_U
            dp_grad += dp_noise / self.hps.batch_size

        return dp_grad

    def __clip(self, x: torch.Tensor, bound: float) -> torch.Tensor:
        return x * torch.min(torch.tensor(1), bound / torch.norm(x))

    def __compute_per_sample_loss_std(self, params: torch.Tensor, buffers: torch.Tensor, sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(self.net, (params, buffers), (batch,))
        loss = self.hps.criterion(predictions, targets)
        return loss

    def __compute_per_sample_loss_adv(self, params: torch.Tensor, buffers: torch.Tensor, sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        loss = functional_call(self.adv_criterion, (params, buffers), (batch, targets))
        return loss

    def __get_batch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        data = next(iter(data_loader))
        batch_data, batch_target = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)

        return batch_data, batch_target

    def __run_schedule(self,curr_epoch: int) -> None:
        decay_step = self.init_hps.step_size * (1 - self.hps.lr_decay_magnitude) / (self.hps.num_epochs - self.hps.decay_epoch_start)
        if curr_epoch > self.hps.decay_epoch_start and self.hps.step_size > self.init_hps.step_size * self.hps.lr_decay_magnitude:
            self.hps.step_size -= decay_step

        if self.attack_type in [AttackType.FGSM, AttackType.PGD] or self.hps.num_burnin_epochs == 0:
            self.hps.eps, self.hps.alpha, self.hps.step_size = self.init_hps.eps, self.init_hps.alpha, self.init_hps.step_size
            return

        # Only for IBP
        delta_eps, delta_alpha = self.init_hps.eps, 1 - self.init_hps.alpha
        increment_eps, increment_alpha = delta_eps / self.hps.eps_warmup_epochs, delta_alpha / self.hps.alpha_warmup_epochs

        if curr_epoch <= self.hps.alpha_warmup_epochs:
            self.hps.alpha -= increment_alpha

        if curr_epoch <= self.hps.eps_warmup_epochs:
            self.hps.eps += increment_eps
            self.hps.step_size = self.init_hps.warmup_step_size
        else:
            self.hps.step_size = self.init_hps.step_size

    def __init_params(self, from_trained: bool = False, path: str = None) -> List[torch.Tensor]:
        init_q = []
        if from_trained:
            self.net.load_state_dict(torch.load(path))
            for param in self.net.parameters():
                init_q.append(copy.deepcopy(param.data))
        else:
            for param in self.net.named_parameters():
                init_vals = torch.empty_like(param[1]).unsqueeze(0)
                init_vals = torch.nn.init.xavier_normal_(init_vals, gain=torch.nn.init.calculate_gain('relu'))
                param[1].data = torch.nn.parameter.Parameter(init_vals.squeeze(0))
                init_q.append(copy.deepcopy(param[1].data))

        return init_q
