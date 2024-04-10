import torch

from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.attack_types import AttackType
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import fgsm_predictive_distrib_attack, ibp_eval
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


def main():

    print(f"Using device: {TORCH_DEVICE}")
    VANILLA_BNN = VanillaBnnLinear().to(TORCH_DEVICE)
    train_data, test_data = load_mnist("./")
    hyperparams_1 = HyperparamsHMC(num_epochs=70, num_burnin_epochs=15, step_size=0.01, lf_steps=600, criterion=torch.nn.CrossEntropyLoss(),
                                   batch_size=100, momentum_std=0.01, run_dp=False, grad_norm_bound=5, dp_sigma=0.1, alpha=0.6, eps=0.132)
    # 82.42%, 32.02%, 6.09% with IBP - 70, 15, 0.6
    # 90.42%, 73.37%, 0.25% with FGSM - 45, 15
    hmc = AdvHamiltonianMonteCarlo(VANILLA_BNN, hyperparams_1, AttackType.IBP)

    samples = hmc.train_mnist_vanilla(train_data)

    hmc.hps.eps = 0.12
    adv_test_set = fgsm_predictive_distrib_attack(hmc.net, hmc.hps, test_data, samples)
    acc_with_average_logits = hmc.test_hmc_with_average_logits(test_data, samples)

    print(f'Accuracy of ADV-HMC-DP with average logit on standard test set: {acc_with_average_logits} %')
    print('------------------- FGSM attack -------------------')
    acc_with_average_logits = hmc.test_hmc_with_average_logits(adv_test_set, samples)
    print(f'Accuracy of ADV-HMC-DP with average logit on adversarially generated test set: {acc_with_average_logits} %')
    print('---------------------------------------------------')

    print('------------------- IBP attack -------------------')
    acc_ibp = ibp_eval(hmc.net, hmc.hps, test_data, samples)
    print(f'Accuracy of ADV-HMC-DP with IBP on standard test set: {acc_ibp} %')
    print('---------------------------------------------------')
    return 0

if __name__ == '__main__':
    main()
