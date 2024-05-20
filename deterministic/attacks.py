import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append('../')

from common.datasets import GenericDataset
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE


def fgsm_test_set_attack(net: VanillaNetLinear, hps: Hyperparameters, test_set: Dataset) -> Dataset:
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    adv_examples_list = []

    for batch_data, batch_target in data_loader:
        batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE), batch_target.to(TORCH_DEVICE)
        batch_data_test.requires_grad = True
        y_hat_test = net(batch_data_test)
        loss = hps.criterion(y_hat_test, batch_target_test)
        net.zero_grad()
        loss.backward()

        # Generate adversarial examples for this batch
        data_grad = batch_data_test.grad.data.clone().detach()
        adv_batch_data = batch_data_test + hps.eps * data_grad.sign()
        # Clamp to valid range [0, 1]
        adv_batch_data = torch.clamp(adv_batch_data, 0, 1)
        adv_examples_list.append(adv_batch_data)

    adv_examples = torch.cat(adv_examples_list, dim=0)
    adv_labels = test_set.targets.to(TORCH_DEVICE)
    adv_dataset = GenericDataset(adv_examples, adv_labels)

    return adv_dataset

def pgd_test_set_attack(net: VanillaNetLinear, hps: Hyperparameters, test_set: Dataset, iterations: int = 10) -> Dataset:
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    adv_examples_list = []

    for batch_data, batch_target in data_loader:
        batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE), batch_target.to(TORCH_DEVICE)
        original_data = batch_data_test.clone().detach()
        for _ in range(iterations):
            batch_data_test.requires_grad = True
            y_hat_test = net(batch_data_test)
            loss = hps.criterion(y_hat_test, batch_target_test)
            net.zero_grad()
            loss.backward()

            # Compute perturbation
            data_grad = batch_data_test.grad.data
            adv_data = batch_data_test + hps.eps * data_grad.sign()

            # Clip perturbation to the epsilon constraint
            eta = torch.clamp(adv_data - original_data, min=-hps.eps, max=hps.eps)
            batch_data_test = torch.clamp(original_data + eta, min=0, max=1).detach_()
        adv_examples_list.append(batch_data_test)

    adv_examples = torch.cat(adv_examples_list, dim=0)
    adv_labels = test_set.targets.to(TORCH_DEVICE)
    adv_dataset = GenericDataset(adv_examples, adv_labels)

    return adv_dataset

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
