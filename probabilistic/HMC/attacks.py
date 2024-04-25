import copy
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')

import wandb

from globals import LOGGER_TYPE, TORCH_DEVICE
from probabilistic.HMC.datasets import GenericDataset
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


def fgsm_predictive_distrib_attack(net: VanillaBnnLinear, hps: HyperparamsHMC, test_set: Dataset, posterior_samples: List[torch.Tensor]) -> Dataset:
    # x_adv_theta = x + eps * sign(E_theta[grad_x log p(y | x, theta)])
    adv_examples_grads_mean = torch.zeros_like(test_set.data, dtype=torch.float32, device=TORCH_DEVICE)
    batch_size = 1000
    data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for sample in posterior_samples:
        net.set_params(sample)

        test_set_adv_grads = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
        for batch_data, batch_target in data_loader:
            batch_data_test, batch_target_test = batch_data.to(TORCH_DEVICE), batch_target.to(TORCH_DEVICE)
            batch_data_test.requires_grad = True
            y_hat_test = net(batch_data_test)
            loss = hps.criterion(y_hat_test, batch_target_test)
            loss.backward()
            batch_adv_grads = copy.deepcopy(batch_data_test.grad.data)
            test_set_adv_grads = torch.cat((test_set_adv_grads, batch_adv_grads), dim=0)
            net.zero_grad()
            batch_data_test.grad.zero_()
        # squeeze dim 1 because image is of shape [1, 28, 28] where 1 is the number of channels, so batch_adv_examples is of shape [batch_size, 1, 28, 28]
        adv_examples_grads_mean += test_set_adv_grads.squeeze(dim=1) / len(posterior_samples)

    adv_inputs = copy.deepcopy(test_set.data.to(TORCH_DEVICE) + hps.eps * torch.sign(adv_examples_grads_mean))
    adv_inputs = torch.clamp(adv_inputs, 0, 1)

    adv_labels = copy.deepcopy(test_set.targets.to(TORCH_DEVICE))
    adv_dataset = GenericDataset(adv_inputs, adv_labels)

    return adv_dataset

def pgd_predictive_distrib_attack(net: VanillaBnnLinear, hps: HyperparamsHMC, test_set: Dataset,
                                  posterior_samples: List[torch.Tensor], iterations: int = 10) -> Dataset:
    # x_adv_theta = PI_S[x + eps * sign(E_theta[grad_x log p(y | x, theta)])]
    curr_dset = GenericDataset(copy.deepcopy(test_set.data.to(TORCH_DEVICE)), copy.deepcopy(test_set.targets.to(TORCH_DEVICE)))
    batch_size = 1000
    for _ in range(iterations):
        adv_examples_grads_mean = torch.zeros_like(test_set.data, dtype=torch.float32, device=TORCH_DEVICE)
        data_loader = DataLoader(curr_dset, batch_size=batch_size, shuffle=False)

        for sample in posterior_samples:
            net.set_params(sample)

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
            adv_examples_grads_mean += test_set_adv_grads.squeeze(dim=1) / len(posterior_samples)

        curr_it_adv_inputs = copy.deepcopy(curr_dset.data.to(TORCH_DEVICE)) + hps.eps * torch.sign(adv_examples_grads_mean)
        delta = torch.clamp(curr_it_adv_inputs - test_set.data.to(TORCH_DEVICE), -hps.eps, hps.eps)
        curr_it_projected_adv_inputs = torch.clamp(test_set.data.to(TORCH_DEVICE) + delta, 0, 1)
        curr_dset = GenericDataset(curr_it_projected_adv_inputs, curr_dset.targets)

    return curr_dset

def ibp_eval(net: VanillaBnnLinear, hps: HyperparamsHMC, test_set: Dataset, posterior_samples: List[torch.Tensor]) -> float:
    avg_worst_case_logits = torch.zeros(len(test_set), net.get_output_size()).to(TORCH_DEVICE)
    batch_size = 1000
    data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    for sample in posterior_samples:
        net.set_params(sample)
        net.eval()

        sample_logits = torch.tensor([]).to(TORCH_DEVICE)
        for data, target in data_loader:
            batch_data_test, y_true_test = data.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
            batch_worst_case_logits = net.get_worst_case_logits(batch_data_test, y_true_test, hps.eps)
            batch_normalized_worst_case_logits = F.softmax(batch_worst_case_logits, dim=1)

            sample_logits = torch.cat((sample_logits, batch_normalized_worst_case_logits), dim=0)

        avg_worst_case_logits += sample_logits / len(posterior_samples)

    print(f"Len posterior samples: {len(posterior_samples)}")
    correct, total = 0, test_set.data.size(0)
    avg_val_of_max_logit = 0
    #* Very basic, but just to be clear
    for i in range(len(test_set)):
        avg_logit = avg_worst_case_logits[i]
        index_of_max_logit = torch.argmax(avg_logit)
        avg_val_of_max_logit += avg_logit[index_of_max_logit]
        if index_of_max_logit == test_set.targets[i]:
            correct += 1

    print(f"Average value of max logit: {avg_val_of_max_logit / total}")
    ibp_robust_acc = 100 * correct / total
    if LOGGER_TYPE == 'wandb':
        wandb.log({'ibp_robust_acc': ibp_robust_acc})

    return ibp_robust_acc
