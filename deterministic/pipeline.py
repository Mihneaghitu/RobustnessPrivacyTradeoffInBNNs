import copy
import sys

sys.path.append('../')

from math import ceil

import torch
from torch.utils.data import DataLoader, Dataset

from common.attack_types import AttackType
from common.dataset_utils import load_mnist
from deterministic.attacks import (fgsm_test_set_attack, ibp_eval,
                                   pgd_test_set_attack)
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import IbpAdversarialLoss, VanillaNetLinear
from globals import TORCH_DEVICE


class PipelineDnn:
    def __init__(self, net: VanillaNetLinear, hyperparams: Hyperparameters, attack_type: AttackType = AttackType.FGSM) -> None:
        self.net = net
        self.hps = hyperparams
        self.attack_type = attack_type
        self.adv_generator, self.adv_criterion = None, self.hps.criterion
        match attack_type:
            case AttackType.FGSM:
                self.adv_generator = self.__gen_fgsm_attack
            case AttackType.PGD:
                self.adv_generator = self.__gen_pgd_attack
            case IBP:
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
            for _, data in enumerate(data_loader):
                self.__run_schedule(itr)
                self.net.zero_grad()
                # ---------------------- Standard forward pass and backprop step ----------------------
                batch_data_train, batch_target_train = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)
                y_hat = self.net(batch_data_train)
                loss = self.hps.criterion(y_hat, batch_target_train)
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

    def __run_schedule(self, itr: int):
        decay_epoch_start = 25
        itr_per_epoch = self.train_data_len // self.hps.batch_size
        decay_itr_start = decay_epoch_start * itr_per_epoch
        decay_itrs = (self.hps.num_epochs - decay_epoch_start) * itr_per_epoch
        delta_lr_decay = (1 - self.hps.lr_decay_magnitude) * self.hps.lr / decay_itrs

        if itr >= decay_itr_start:
            self.hps.lr -= delta_lr_decay

        if self.attack_type == AttackType.IBP:
            warmup_epoch_start, warmup_epoch_end = 5, 15
            warmup_itr_start, warmup_itr_end = warmup_epoch_start * itr_per_epoch, warmup_epoch_end * itr_per_epoch
            num_warmup_itrs = warmup_itr_end - warmup_itr_start
            delta_warmup_eps, delta_warmup_alpha = self.copy_eps / num_warmup_itrs, (1 - self.copy_alpha) / num_warmup_itrs
            if warmup_itr_start <= itr < warmup_itr_end:
                self.hps.eps += delta_warmup_eps
                self.hps.alpha -= delta_warmup_alpha


def run_pipeline(test_only: bool = False, save_model: bool = False):
    net = VanillaNetLinear().to(TORCH_DEVICE)
    curr_dir = __file__.rsplit('/', 2)[0]
    hyperparams = Hyperparameters(num_epochs=20,
                                  lr=0.005,
                                  batch_size=60,
                                  criterion=torch.nn.CrossEntropyLoss(),
                                  lr_decay_magnitude=0.25,
                                  alpha=1,
                                  eps=0.1)

    pipeline = PipelineDnn(net, hyperparams, AttackType.FGSM)
    if not test_only:
        train, test = load_mnist()
        pipeline.train_mnist_vanilla(train)
        if save_model:
            torch.save(pipeline.net.state_dict(), curr_dir + '/vanilla_network.pt')
    else:
        print("Testing only...")
        net = VanillaNetLinear()
        net.load_state_dict(torch.load(curr_dir + 'vanilla_network.pt'))

    std_acc = pipeline.test_mnist_vanilla(test)
    print(f'Standard accuracy of the network on the 10000 test images: {std_acc}%')
    fgsm_test_set = fgsm_test_set_attack(net, hyperparams, test)
    fgsm_acc = pipeline.test_mnist_vanilla(fgsm_test_set)
    print(f'Accuracy of the network using FGSM adversarial test set: {fgsm_acc}%')
    pgd_test_set = pgd_test_set_attack(net, hyperparams, test)
    pgd_acc = pipeline.test_mnist_vanilla(pgd_test_set)
    print(f'Accuracy of the network using PGD adversarial test set: {pgd_acc}%')
    ibp_acc = ibp_eval(net, hyperparams, test)
    print(f'Accuracy of the network using IBP adversarial test set: {ibp_acc}%')
