import copy
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append('../')

import wandb

from common.datasets import GenericDataset
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import VanillaNetLinear
from globals import LOGGER_TYPE, TORCH_DEVICE


def fgsm_test_set_attack(net: VanillaNetLinear, hps: Hyperparameters, test_set: Dataset) -> Dataset:
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    net.eval()
    adv_grads = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)

    for batch_data, batch_target in data_loader:
        batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE), batch_target.to(TORCH_DEVICE)
        batch_data_test.requires_grad = True
        y_hat_test = net(batch_data_test)
        loss = hps.criterion(y_hat_test, batch_target_test)
        loss.backward()
        batch_adv_grads = copy.deepcopy(batch_data_test.grad.data)
        adv_grads = torch.cat((adv_grads, batch_adv_grads), dim=0)
        net.zero_grad()
        batch_data_test.grad.zero_()
    # squeeze dim 1 because image is of shape [1, 28, 28] where 1 is the number of channels, so batch_adv_examples is of shape [batch_size, 1, 28, 28]
    adv_grads = adv_grads.squeeze(dim=1)

    adv_inputs = copy.deepcopy(test_set.data.to(TORCH_DEVICE) + hps.eps * torch.sign(adv_grads))
    adv_inputs = torch.clamp(adv_inputs, 0, 1)
    adv_labels = copy.deepcopy(test_set.targets.to(TORCH_DEVICE))
    adv_dataset = GenericDataset(adv_inputs, adv_labels)

    return adv_dataset

def pgd_test_set_attack(net: VanillaNetLinear, hps: Hyperparameters, test_set: Dataset, iterations: int = 10) -> Dataset:
    curr_dset = GenericDataset(copy.deepcopy(test_set.data.to(TORCH_DEVICE)), copy.deepcopy(test_set.targets.to(TORCH_DEVICE)))
    net.eval()

    for _ in range(iterations):
        adv_examples_grads = torch.zeros_like(test_set.data, dtype=torch.float32, device=TORCH_DEVICE)
        data_loader = DataLoader(curr_dset, batch_size=1000, shuffle=False)

        test_set_adv_grads = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
        for batch_data, batch_target in data_loader:
            batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE, dtype=torch.float32), batch_target.to(TORCH_DEVICE)
            batch_data_test.requires_grad = True
            y_hat_test = net(batch_data_test)
            loss = hps.criterion(y_hat_test, batch_target_test)
            loss.backward()
            batch_adv_grads = copy.deepcopy(batch_data_test.grad.data)
            test_set_adv_grads = torch.cat((test_set_adv_grads, batch_adv_grads), dim=0)
            net.zero_grad()
            batch_data_test.grad.zero_()
        # squeeze dim 1 because image is of shape [1, 28, 28] where 1 is the number of channels, so batch_adv_examples is of shape [batch_size, 1, 28, 28]
        adv_examples_grads += test_set_adv_grads.squeeze(dim=1)

        curr_it_adv_inputs = copy.deepcopy(curr_dset.data.to(TORCH_DEVICE)) + hps.eps * torch.sign(adv_examples_grads)
        delta = torch.clamp(curr_it_adv_inputs - test_set.data.to(TORCH_DEVICE), -hps.eps, hps.eps)
        curr_it_projected_adv_inputs = torch.clamp(test_set.data.to(TORCH_DEVICE) + delta, 0, 1)
        curr_dset = GenericDataset(curr_it_projected_adv_inputs, curr_dset.targets)

    return curr_dset

def ibp_eval(net: VanillaNetLinear, hps: Hyperparameters, test_set: Dataset) -> float:
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=2)
    net.eval()

    worst_case_logits = torch.tensor([]).to(TORCH_DEVICE)
    for data, target in data_loader:
        batch_data_test, y_true_test = data.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
        batch_worst_case_logits = net.get_worst_case_logits(batch_data_test, y_true_test, hps.eps)
        batch_normalized_worst_case_logits = F.softmax(batch_worst_case_logits, dim=1)

        worst_case_logits = torch.cat((worst_case_logits, batch_normalized_worst_case_logits), dim=0)

    correct, total = 0, test_set.data.size(0)
    #* Very basic, but just to be clear
    for i in range(len(test_set)):
        if torch.argmax(worst_case_logits[i]) == test_set.targets[i]:
            correct += 1

    ibp_robust_acc = 100 * correct / total

    return ibp_robust_acc
