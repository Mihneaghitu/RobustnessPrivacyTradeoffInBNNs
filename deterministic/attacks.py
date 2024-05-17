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
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    adv_examples = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)

    for batch_data, batch_target in data_loader:
        batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE), batch_target.to(TORCH_DEVICE)
        batch_data_test.requires_grad = True
        y_hat_test = net(batch_data_test)
        loss = hps.criterion(y_hat_test, batch_target_test)
        loss.backward()
        data_grad = copy.deepcopy(batch_data_test.grad.data)
        curr_adv_example = torch.clamp(batch_data_test + hps.eps * data_grad.sign(), 0, 1)
        adv_examples = torch.cat((adv_examples, curr_adv_example), dim=0)
        net.zero_grad()
    # squeeze dim 1 because image is of shape [1, 28, 28] where 1 is the number of channels, so batch_adv_examples is of shape [batch_size, 1, 28, 28]
    adv_examples = adv_examples.squeeze(dim=1)

    adv_labels = copy.deepcopy(test_set.targets.to(TORCH_DEVICE))
    adv_dataset = GenericDataset(adv_examples, adv_labels)

    return adv_dataset

def pgd_test_set_attack(net: VanillaNetLinear, hps: Hyperparameters, test_set: Dataset, iterations: int = 10) -> Dataset:
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    adv_inputs = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, batch_target in data_loader:
        batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE), batch_target.to(TORCH_DEVICE)
        curr_batch_adv_examples = batch_data_test.clone().detach()
        for _ in range(iterations):
            curr_batch_adv_examples.requires_grad = True
            y_hat_test = net(curr_batch_adv_examples)
            loss = hps.criterion(y_hat_test, batch_target_test)
            loss.backward()

            new_it_adv_examples = copy.deepcopy(curr_batch_adv_examples) + hps.eps * curr_batch_adv_examples.grad.data.sign()
            delta = torch.clamp(new_it_adv_examples - batch_data_test, -hps.eps, hps.eps)
            curr_it_projected_adv_examples = torch.clamp(batch_data_test + delta, 0, 1)
            curr_batch_adv_examples = curr_it_projected_adv_examples.clone().detach()
            net.zero_grad()
        adv_inputs = torch.cat((adv_inputs, curr_batch_adv_examples), dim=0)

    adv_dset = GenericDataset(adv_inputs, copy.deepcopy(test_set.targets))
    return adv_dset

def ibp_eval(net: VanillaNetLinear, hps: Hyperparameters, test_set: Dataset) -> float:
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
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
