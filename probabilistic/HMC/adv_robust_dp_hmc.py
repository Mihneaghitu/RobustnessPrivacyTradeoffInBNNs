import copy
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
import torchvision
import wandb
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')

from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.attack_types import AttackType
from probabilistic.HMC.attacks import fgsm_predictive_distrib_attack
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import IbpAdversarialLoss, VanillaBnnLinear


class AdvHamiltonianMonteCarlo:
    def __init__(self, net: VanillaBnnLinear, hyperparameters: HyperparamsHMC, attack_type: AttackType = AttackType.FGSM):
        self.net = net
        self.hps = hyperparameters
        self.attack_type = attack_type
        self.adv_generator = None
        self.adv_criterion = None
        self.eps_schedule = None
        match attack_type:
            case AttackType.FGSM:
                self.adv_generator = self.__gen_fgsm_adv_examples
                self.adv_criterion = torch.nn.CrossEntropyLoss()
                self.eps_schedule = lambda _: self.hps.eps
            case AttackType.PGD:
                self.adv_generator = self.__gen_pgd_adv_examples
                self.adv_criterion = torch.nn.CrossEntropyLoss()
                self.eps_schedule = lambda _: self.hps.eps
            case AttackType.IBP:
                self.adv_generator = self.__gen_ibp_adv_examples
                self.adv_criterion = IbpAdversarialLoss(net, torch.nn.CrossEntropyLoss(), self.hps.eps)
                # increases fast at the beginning, then slowly, saturating at eps
                # self.eps_schedule = lambda epoch: (1 - np.exp(- epoch / 10)) * self.hps.eps
                # linear schedule between start and end, based on epoch
                self.eps_schedule = lambda start, end, epoch, eps: eps * (epoch - start) / (end - start)
                self.alpha_schedule = lambda start, end, epoch, alpha: alpha * (end - epoch) / (end - start)


    def train_mnist_vanilla(self, train_set: torchvision.datasets.mnist) -> List[torch.tensor]:
        # Just don't ask
        print_freq = self.hps.lf_steps - 1 # 500 // (self.hps.batch_size / 64)

        data_loader = DataLoader(train_set, batch_size=self.hps.batch_size, shuffle=True)
        posterior_samples = []

        # Initialize the parameters with a standard normal --
        # q -> current net params, current q -> start net params
        current_q = []
        for param in self.net.named_parameters():
            init_vals = torch.normal(mean=0.0, std=0.1, size=tuple(param[1].shape)).to(TORCH_DEVICE)
            param[1].data = torch.nn.parameter.Parameter(init_vals)
            current_q.append(copy.deepcopy(param[1].data))

        running_loss_ce, running_loss_ce_adv = 0.0, 0.0
        copy_eps, copy_alpha = self.hps.eps, self.hps.alpha
        itr = 0
        for epoch in range(self.hps.num_epochs):
            losses, p = [], []
            for param in self.net.parameters():
                p.append(dist.Normal(0, self.hps.momentum_std).sample(param.shape).to(TORCH_DEVICE))
            current_p = copy.deepcopy(p)

            # ------- half step for momentum -------
            closs, closs_adv = self.__p_update(data_loader, p, self.hps.step_size / 2)
            running_loss_ce += closs.item()
            running_loss_ce_adv += closs_adv.item()
            for i in range(self.hps.lf_steps):
                if self.attack_type == AttackType.IBP:
                    if itr == 22000:
                        self.hps.step_size /= 5
                    #* increase slowly (for IBP), as mentioned by Gowal et al. 2018 (IBP paper)
                    if itr <= 9000:
                        self.hps.eps = 0.0
                        self.hps.alpha = 1.0
                    elif 9000 < itr < 20000:
                        self.hps.eps += copy_eps / 11000
                        self.hps.alpha -= (1 - copy_alpha) / 11000
                    else:
                        self.hps.eps = copy_eps
                        self.hps.alpha = copy_alpha
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
                itr +=1
            # wandb.log({'cross_entropy_loss': losses[-1]})
            # wandb.log({'epoch': epoch + 1})

            # -------------- Final half step for momentum --------------
            closs, closs_adv = self.__p_update(data_loader, p, self.hps.step_size / 2)
            for idx, p_val in enumerate(p):
                p[idx] = -p_val

            # metropolis-hastings acceptance step
            q = self.net.get_params()
            initial_energy = self.__get_energy(current_q, current_p, self.hps.criterion, data_loader)
            end_energy = self.__get_energy(q, p, self.hps.criterion, data_loader)
            acceptance_prob = min(1, torch.exp(end_energy - initial_energy))
            print(f'Acceptance probability: {acceptance_prob}')
            # wandb.log({'acceptance_probability': acceptance_prob})

            if dist.Uniform(0, 1).sample().to(TORCH_DEVICE) < acceptance_prob:
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

        print(f"Len posterior samples: {len(posterior_samples)}")
        correct, total = 0, test_set.targets.size(0)
        avg_val_of_max_logit = 0
        for i in range(test_set.targets.size(0)):
            avg_logit = average_logits[i]
            softmaxed_avg_logit = F.softmax(avg_logit, dim=0)
            index_of_max_logit = torch.argmax(avg_logit)
            avg_val_of_max_logit += softmaxed_avg_logit[index_of_max_logit].item()
            if index_of_max_logit == test_set.targets[i]:
                correct += 1

        print(f"Average value of max logit: {avg_val_of_max_logit / total}")
        # wandb.log({'accuracy_with_average_logits': 100 * correct / total})

        return 100 * correct / total

    # ---------------------------------------------------------
    # -------------------- Helper functions -------------------
    # ---------------------------------------------------------

    def __p_update(self, data_loader: torch.utils.data.DataLoader, p: list, lf_step: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_data, batch_target = self.__get_batch(data_loader)
        closs = self.__get_nll_loss(self.hps.criterion, (batch_data, batch_target), adv=False)
        self.__advance_momentum(p, self.hps.alpha * lf_step, adv=False)
        closs_adv = self.__get_nll_loss(self.adv_criterion, (batch_data, batch_target), adv=True)
        self.__advance_momentum(p, (1 - self.hps.alpha) * lf_step, adv=True)

        return closs, closs_adv

    def __advance_momentum(self, p: list, eps: float, adv=False) -> None:
        prior_loss = torch.tensor(0.0).requires_grad_(True).to(TORCH_DEVICE)
        prior_grad = torch.tensor(0.0).requires_grad_(True).to(TORCH_DEVICE)
        for idx, param in enumerate(self.net.parameters()):
            ll_grad = param.grad
            if not adv:
                prior_loss = prior_loss + torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
                prior_grad = torch.autograd.grad(outputs=prior_loss, inputs=param)[0]
            potential_energy_grad = copy.deepcopy(ll_grad) + prior_grad
            if self.hps.run_dp:
                # clip the gradient norm (first term) and add noise (second term)
                potential_energy_grad /= max(1, torch.norm(potential_energy_grad) / self.hps.grad_norm_bound)
                dp_noise = dist.Normal(0, self.hps.dp_sigma * self.hps.grad_norm_bound).sample(potential_energy_grad.shape).to(TORCH_DEVICE)
                # add the mean noise across the batch to grad_U
                potential_energy_grad += dp_noise / self.hps.batch_size
            p[idx] = self.net.update_param(p[idx], potential_energy_grad, eps)

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

        closs.backward()

        return closs

    def __gen_pgd_adv_examples(self, batch_input: torch.Tensor, batch_target: torch.Tensor, bound=0.3, eps=0.05, iters=10):
        curr_input = copy.deepcopy(batch_input)
        for _ in range(iters) :
            curr_input.requires_grad = True
            y_hat = self.net(curr_input)

            self.net.zero_grad()
            loss = self.hps.criterion(y_hat, batch_target)
            loss.backward()

            adv_examples = curr_input + eps * torch.sign(curr_input.grad.data)
            delta = torch.clamp(adv_examples - batch_input, min=-bound, max=bound)
            adv_examples = torch.clamp(batch_input + delta, min=0, max=1).detach_()

        return adv_examples

    def __gen_fgsm_adv_examples(self, batch_input: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        batch_input.requires_grad = True
        y_hat = self.net(batch_input)
        loss = self.hps.criterion(y_hat, batch_target)
        loss.backward()
        input_grads = copy.deepcopy(batch_input.grad.data)
        adv_examples = copy.deepcopy(batch_input)
        adv_examples = adv_examples + self.hps.eps * torch.sign(input_grads)
        clamped_adv_examples = torch.clamp(adv_examples, 0, 1)
        batch_input.grad.zero_()
        batch_input.requires_grad = False
        self.net.zero_grad()

        return clamped_adv_examples

    def __gen_ibp_adv_examples(self, batch_input: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        #* This looks like a dummy function, and it is. Nevertheless, it's necessary for the sake of consistency
        return batch_input

    def __get_energy(self, q: list, p: list, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        # save the current parameters
        start_params = self.net.get_params()

        # first update the nk params
        for idx, param in enumerate(self.net.parameters()):
            with torch.no_grad():
                param.copy_(copy.deepcopy(q[idx]))

        # compute the potential energy
        batch_data, batch_target = self.__get_batch(data_loader)
        closs = criterion(self.net(batch_data), batch_target)
        adv_ex = self.adv_generator(batch_data, batch_target)
        closs_adv = criterion(self.net(adv_ex), batch_target)
        prior_loss = torch.tensor(0.0).to(TORCH_DEVICE)
        for idx, param in enumerate(self.net.parameters()):
            prior_loss += torch.neg(torch.mean(dist.Normal(self.hps.prior_mu, self.hps.prior_std).log_prob(param)))
        potential_energy = self.hps.alpha * closs + (1 - self.hps.alpha) * closs_adv + prior_loss

        # compute the kinetic energy
        kinetic_energy = torch.tensor(0.0).to(TORCH_DEVICE)
        for idx, p_val in enumerate(p):
            kinetic_energy = kinetic_energy + torch.sum(p_val * p_val) / 2

        # reset the parameters
        self.net.set_params(start_params)
        return potential_energy + kinetic_energy

    def __get_batch(self, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        data = next(iter(data_loader))
        batch_data, batch_target = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)

        return batch_data, batch_target

#@ RESULTS FOR EXPERIMENT 1: eps train 0.1, eps test 0.05 (NON-ROBUST NON-DP MODEL VS. ADVERSARIALLY ROBUST NON-DP MODEL):
#! 1. NORMALLY TRAINED MODEL (20 epochs, 8 burnin-epochs) :
#*    On normal test set => 74.47%
#*    On adversarially generated test set => 44.23%
#! 2. NORMALLY TRAINED MODEL (45 epochs, 8 burnin-epochs) :
#*    On normal test set => 84.17%
#*    On adversarially generated test set => 54.26%
#! 3. ADVERSARIALLY TRAINED MODEL (20 epochs, 8 burnin-epochs):
#*    On normal test set => 64.64%
#*    On adversarially generated test set => 41.68%
#! 4. ADVERSARIALLY TRAINED MODEL (45 epochs, 8 burnin-epochs):
#*    On normal test set => 75.35%
#*    On adversarially generated test set => 58.06%
# -> learns pretty slow adversarially
# ------------------------------------------------------------------------------------------------------------------
#@ RESULTS FOR EXPERIMENT 2: eps train 0.1, eps test 0.05  (NON-ROBUST DP MODEL VS. ADVERSARIALLY ROBUST DP MODEL):
#! 1. NORMALLY TRAINED MODEL (25 epochs, 8 burnin-epochs, grad_bound 5, dp sigma 0.1):
#*    On normal test set => 79.27%
#*    On adversarially generated test set => 49.07%
#! 2. NORMALLY TRAINED MODEL (50 epochs, 10 burnin-epochs):
#*    On normal test set => 85.4%
#*    On adversarially generated test set => 56.48%
#! 3. ADVERSARIALLY TRAINED MODEL (25 epochs, 8 burnin-epochs, grad_bound 5, dp sigma 0.1):
#*    On normal test set => 63.98%
#*    On adversarially generated test set => 42.84%
#! 4. ADVERSARIALLY TRAINED MODEL (50 epochs, 10 burnin-epochs):
#*    On normal test set => 75.46%
#*    On adversarially generated test set => 57.03%

#? ------------------ SOME CONCLUSIONS (MAYBE) ------------------
#? 1. The adversarially trained model is more robust to adversarial examples than the normally trained model.
#? 2. DP naturally makes the model more robust to adversarial examples (perhaps due to random perturbation)
#? 3. !!! DP also helps adversarially trained models to be more robust to adversarial examples (for the same reasons?)