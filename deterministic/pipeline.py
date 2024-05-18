import copy
import sys

sys.path.append('../')

from math import ceil

import torch
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Dataset

from common.attack_types import AttackType
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import IbpAdversarialLoss, VanillaNetLinear
from globals import TORCH_DEVICE


class PipelineDnn:
    def __init__(self, net: VanillaNetLinear, hyperparams: Hyperparameters, attack_type: AttackType = AttackType.FGSM) -> None:
        self.net = net
        self.hps = hyperparams
        self.init_hps = copy.deepcopy(hyperparams)
        self.attack_type = attack_type
        self.adv_generator, self.adv_criterion = None, self.hps.criterion
        match attack_type:
            case AttackType.FGSM:
                self.adv_generator = self.__gen_fgsm_attack
            case AttackType.PGD:
                self.adv_generator = self.__gen_pgd_attack
            case AttackType.IBP:
                self.adv_generator = self.__gen_ibp_attack
                self.adv_criterion = IbpAdversarialLoss(self.net, self.hps.criterion, self.hps.eps)
                # for warmup
                self.copy_eps, self.copy_alpha = self.hps.eps, self.hps.alpha
                self.hps.eps, self.hps.alpha = 0, 1
        self.train_data_len = 0

    def train_mnist_vanilla(self, train_data: Dataset) -> None:
        self.net.train()
        data_loader = DataLoader(train_data, batch_size=self.hps.batch_size, shuffle=True)
        num_batches_per_epoch = int(ceil(train_data.data.shape[0] // self.hps.batch_size))
        self.train_data_len = train_data.data.shape[0]

        # Initialize the parameters with a standard normal
        for param in self.net.named_parameters():
            if 'weight' in param[0]:
                init_vals = torch.normal(mean=0.0, std=0.1, size=tuple(param[1].shape)).to(TORCH_DEVICE)
                param[1].data = torch.nn.parameter.Parameter(init_vals)

        running_loss_std, running_loss_adv = 0.0, 0.0
        itr = 1
        for epoch in range(self.hps.num_epochs):
            losses  = []
            self.__run_lr_schedule(epoch + 1)
            for _, data in enumerate(data_loader):
                self.__run_warmup_schedule(itr)
                self.net.zero_grad()
                # ---------------------- Standard forward pass and backprop step ----------------------
                batch_data_train, batch_target_train = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)
                y_hat = self.net(batch_data_train)
                loss = self.hps.criterion(y_hat, batch_target_train)
                if self.hps.run_dp:
                    self.__privatize(batch_data_train, batch_target_train, adv=False)
                else:
                    loss.backward()
                with torch.no_grad():
                    for param in self.net.parameters():
                        new_val = self.net.update_param(param, param.grad, self.hps.alpha * self.hps.lr)
                        param.copy_(new_val)
                # --------------------------------------------------------------------------------------
                self.net.zero_grad()
                # --------------------- Adversarial forward pass and backprop step ---------------------
                adv_batch_data = self.adv_generator(batch_data_train, batch_target_train)
                y_hat_adv = self.net(adv_batch_data) if self.attack_type != AttackType.IBP else adv_batch_data
                closs = self.adv_criterion(y_hat_adv, batch_target_train)
                if self.hps.run_dp:
                    self.__privatize(batch_data_train, batch_target_train, adv=False)
                else:
                    closs.backward()
                with torch.no_grad():
                    for param in self.net.parameters():
                        new_val = self.net.update_param(param, param.grad, (1 - self.hps.alpha) * self.hps.lr)
                        param.copy_(new_val)
                # --------------------------------------------------------------------------------------

                losses.append((loss.item(), closs.item()))
                running_loss_std += loss.item()
                running_loss_adv += closs.item()
                itr += 1
            print(f'[epoch {epoch + 1}] average standard loss: {running_loss_std / (num_batches_per_epoch)}')
            print(f'[epoch {epoch + 1}] average adversarial loss: {running_loss_adv / (num_batches_per_epoch)}')
            running_loss_std, running_loss_adv = 0.0, 0.0

    def test_mnist_vanilla(self, test_data: Dataset):
        self.net.eval()
        data_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        losses, correct, total = [], 0, 0

        for batch_data, batch_target in data_loader:
            batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE), batch_target.to(TORCH_DEVICE)
            y_hat = self.net(batch_data_test)
            loss = self.hps.criterion(y_hat, batch_target_test)
            losses.append(loss.item())
            _, predicted = torch.max(y_hat, 1)
            total += batch_target_test.size(0)
            correct += (predicted == batch_target_test).sum().item()

        return 100 * correct / total

    def __privatize(self, batch_data: torch.Tensor, batch_target: torch.Tensor, adv: bool = False) -> torch.Tensor:
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

    def __clip(self, x: torch.Tensor, bound: float) -> torch.Tensor:
        return x * torch.min(torch.tensor(1), bound / torch.norm(x))

    def __gen_fgsm_attack(self, batch_data: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        copy_batch_data = copy.deepcopy(batch_data.detach().clone())
        copy_batch_data.requires_grad = True
        y_hat = self.net(copy_batch_data)
        loss = self.hps.criterion(y_hat, batch_target)
        loss.backward()
        input_grad = copy.deepcopy(copy_batch_data.grad.data)

        adv_batch_data = copy.deepcopy(batch_data.detach().clone()) + self.hps.eps * torch.sign(input_grad)
        adv_batch_data = torch.clamp(adv_batch_data, 0, 1)

        self.net.zero_grad()

        return adv_batch_data

    def __gen_pgd_attack(self, batch_data: torch.Tensor, batch_target: torch.Tensor, iterations: int = 10) -> torch.Tensor:
        curr_input = copy.deepcopy(batch_data)
        for _ in range(iterations) :
            curr_input.requires_grad = True
            y_hat = self.net(curr_input)

            self.net.zero_grad()
            loss = self.hps.criterion(y_hat, batch_target)
            loss.backward()

            cur_it_projected_adv_examples = copy.deepcopy(curr_input) + self.hps.eps * torch.sign(curr_input.grad.data)
            delta = torch.clamp(cur_it_projected_adv_examples - batch_data, min=-self.hps.eps, max=self.hps.eps)
            cur_it_projected_adv_examples = torch.clamp(batch_data + delta, min=0, max=1).detach_()
            curr_input = copy.deepcopy(cur_it_projected_adv_examples)

        return cur_it_projected_adv_examples

    def __gen_ibp_attack(self, batch_data: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return batch_data

    def __run_warmup_schedule(self, itr: int):
        if self.attack_type == AttackType.IBP:
            delta_warmup_eps = self.copy_eps / self.hps.eps_warmup_itrs
            delta_warmup_alpha = (1 - self.copy_alpha) / self.hps.alpha_warmup_itrs
            if self.hps.warmup_itr_start < itr <= self.hps.warmup_itr_start + self.hps.eps_warmup_itrs:
                self.hps.eps += delta_warmup_eps
            if self.hps.warmup_itr_start < itr <= self.hps.warmup_itr_start + self.hps.alpha_warmup_itrs:
                self.hps.alpha -= delta_warmup_alpha

    def __run_lr_schedule(self, epoch: int):
        if epoch > self.hps.decay_epoch_start:
            delta = (1 - self.hps.lr_decay_magnitude) * self.hps.lr / (self.hps.num_epochs - self.hps.decay_epoch_start)
            self.hps.lr -= delta
