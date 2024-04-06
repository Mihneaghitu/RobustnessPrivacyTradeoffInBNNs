import copy
import sys
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')

from globals import TORCH_DEVICE
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
        net.eval()

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
    # NOTE: be careful here: apparently a dataset needs to be on the cpu so that the DataLoader can work with it
    # NOTE: otherwise CUDA just freaks out: run "CUDA_LAUNCH_BLOCKING=1 <your_program>.py' to see the error
    adv_dataset = GenericDataset(adv_inputs, adv_labels)
    return adv_dataset

def pgd_predictive_distrib_attack(net: VanillaBnnLinear, hps: HyperparamsHMC, test_set: Dataset, posterior_samples: List[torch.Tensor],
                                  bound: float = 0.3, iterations: int = 10) -> Dataset:
    # x_adv_theta = PI_S[x + eps * sign(E_theta[grad_x log p(y | x, theta)])]
    curr_dset = GenericDataset(copy.deepcopy(test_set.data.to(TORCH_DEVICE)), copy.deepcopy(test_set.targets.to(TORCH_DEVICE)))
    for _ in range(iterations):
        adv_examples_grads_mean = torch.zeros_like(test_set.data, dtype=torch.float32, device=TORCH_DEVICE)
        batch_size = 1000
        data_loader = DataLoader(curr_dset, batch_size=batch_size, shuffle=False)

        for sample in posterior_samples:
            net.set_params(sample)
            net.eval()

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

        curr_it_adv_inputs = copy.deepcopy(curr_dset.data.to(TORCH_DEVICE) + hps.eps * torch.sign(adv_examples_grads_mean))
        delta = torch.clamp(curr_it_adv_inputs - curr_dset.data.to(TORCH_DEVICE), -bound, bound)
        curr_it_projected_adv_inputs = torch.clamp(curr_dset.data.to(TORCH_DEVICE) + delta, 0, 1)
        curr_dset = GenericDataset(curr_it_projected_adv_inputs, curr_dset.targets)

    return curr_dset

# check if c^T * net(x) + d <= 0
def verify_predictive_distrib_spec(c: torch.Tensor, d: torch.Tensor, net: VanillaBnnLinear, hps: HyperparamsHMC,
                                   test_set: Dataset, posterior_samples: List[torch.Tensor]) -> float:
    pass