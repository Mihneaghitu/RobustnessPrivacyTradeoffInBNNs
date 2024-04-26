import torch

from dataset_utils import get_marginal_distributions, load_mnist
from deterministic.membership_inference_dnn import MembershipInferenceAttack
from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE
from probabilistic.attack_types import AttackType
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import (fgsm_predictive_distrib_attack,
                                       ibp_eval, pgd_predictive_distrib_attack)
from probabilistic.HMC.datasets import GenericDataset
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.membership_inference_bnn import \
    MembershipInferenceAttackBnn
from probabilistic.HMC.vanilla_bnn import (VanillaBnnFashionMnist,
                                           VanillaBnnMnist)


def main():
    #* Seed is set in globals.py
    # membership_inference_dnn_experiment()
    # membership_inference_bnn_experiment()
    adversarial_robustness_experiment()
    return 0

def adversarial_robustness_experiment():
    print(f"Using device: {TORCH_DEVICE}")
    # vanilla_bnn = VanillaBnnMnist().to(TORCH_DEVICE)
    vanilla_bnn = VanillaBnnMnist().to(TORCH_DEVICE)
    train_data, test_data = load_mnist("./")
    hyperparams_1 = HyperparamsHMC(num_epochs=45, num_burnin_epochs=20, step_size=0.02, warmup_step_size=0.2, lf_steps=120,
                                   criterion=torch.nn.CrossEntropyLoss(), batch_size=1000, momentum_std=0.01, alpha=0.5, eps=0.075)
    hmc = AdvHamiltonianMonteCarlo(vanilla_bnn, hyperparams_1, attack_type=AttackType.IBP)

    samples = hmc.train_mnist_vanilla(train_data)

    hmc.hps.eps = 0.0675
    adv_test_set_pgd = pgd_predictive_distrib_attack(hmc.net, hmc.hps, test_data, samples)
    adv_test_set_fgsm = fgsm_predictive_distrib_attack(hmc.net, hmc.hps, test_data, samples)

    print('------------------- Normal Accuracy -------------------')
    acc_standard = hmc.test_hmc_with_average_logits(test_data, samples)
    print(f'Accuracy of ADV-HMC-DP with average logit on standard test set: {acc_standard} %')
    print('-------------------------------------------------------')

    print('------------------- FGSM attack --------------------')
    acc_fgsm = hmc.test_hmc_with_average_logits(adv_test_set_fgsm, samples)
    print(f'Accuracy of ADV-HMC-DP with average logit on FGSM adversarial test set: {acc_fgsm} %')
    print('----------------------------------------------------')

    print('------------------- PGD attack -------------------')
    acc_pgd = hmc.test_hmc_with_average_logits(adv_test_set_pgd, samples)
    print(f'Accuracy of ADV-HMC-DP with average logit on PGD adversarial test set: {acc_pgd} %')
    print('---------------------------------------------------')

    print('------------------- IBP attack -------------------')
    acc_ibp = ibp_eval(hmc.net, hmc.hps, test_data, samples)
    print(f'Accuracy of ADV-HMC-DP with IBP on standard test set: {acc_ibp} %')
    print('---------------------------------------------------')

def membership_inference_dnn_experiment():
    # -------------- Hyperparams Values --------------
    batch_size, num_epochs, lr = 100, 20, 0.001
    # ------------------------------------------------
    train_data, _ = load_mnist()
    moments = get_marginal_distributions(train_data)
    target_network = VanillaNetLinear().to(TORCH_DEVICE)
    target_network.load_state_dict(torch.load('vanilla_network.pt'))
    membership_inference_attack = MembershipInferenceAttack(target_network, target_network.get_output_size(), moments)
    print("Training shadow models...")
    input_for_attack_models = membership_inference_attack.train_shadow_models(batch_size, num_epochs, lr)
    print("Finished training shadow models.")
    # ------------------------------------------------
    print("Training attack models...")
    membership_inference_attack.train_attack_models(input_for_attack_models, batch_size, num_epochs, lr)
    print("Finished training attack models.")
    # ------------------------------------------------
    num_test_samples = len(train_data) // 2
    # take half of those from train_data (without targets)
    pos_attack_models_test_data = train_data.data[:num_test_samples // 2].detach().clone().to(TORCH_DEVICE)
    sample_shape = pos_attack_models_test_data.shape[1:]
    neg_attack_models_test_data = []
    for _ in range(num_test_samples // 2):
        neg_sample = torch.normal(moments[0], moments[1])
        neg_sample = torch.where(neg_sample < 0, torch.zeros_like(neg_sample), torch.ones_like(neg_sample)).reshape(sample_shape)
        neg_attack_models_test_data.append(neg_sample.tolist())
    neg_attack_models_test_data = torch.tensor(neg_attack_models_test_data).to(TORCH_DEVICE)
    attack_models_test_data = torch.cat((pos_attack_models_test_data, neg_attack_models_test_data), dim=0)
    attack_models_test_targets = torch.cat((torch.ones(num_test_samples // 2, 1), torch.zeros(num_test_samples // 2, 1)), dim=0)
    attack_models_dset = GenericDataset(attack_models_test_data, attack_models_test_targets)

    print("Testing attack models...")
    membership_inference_attack.test_attack_models_dnn(attack_models_dset)
    print("Finished testing attack models.")
    #! Accuracy: 62.71%, 7000 synthetic records

def membership_inference_bnn_experiment():
    print("Training Bayesian Neural Network using HMC...")
    train_data, _ = load_mnist()
    moments = get_marginal_distributions(train_data)
    target_network = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=20, num_burnin_epochs=5, step_size=0.01, lf_steps=120, criterion=torch.nn.CrossEntropyLoss(),
                                 batch_size=100, momentum_std=0.01)
    hmc = HamiltonianMonteCarlo(target_network, hyperparams)
    posterior_samples = hmc.train_mnist_vanilla(train_data)
    accuracy = hmc.test_hmc_with_average_logits(train_data, posterior_samples)
    print(f"Accuracy of BNN with average logits on training set: {accuracy} %")


    batch_size, num_epochs, lr = 100, 20, 0.001
    membership_inference_attack = MembershipInferenceAttackBnn(target_network, target_network.get_output_size(), moments, posterior_samples)
    print("Training shadow models...")
    input_for_attack_models = membership_inference_attack.train_shadow_models(batch_size, num_epochs, lr)
    print("Finished training shadow models.")
    # ------------------------------------------------
    print("Training attack models...")
    membership_inference_attack.train_attack_models(input_for_attack_models, batch_size, num_epochs, lr)
    print("Finished training attack models.")
    # ------------------------------------------------
    num_test_samples = len(train_data) // 2
    # take half of those from train_data (without targets)
    pos_attack_models_test_data = train_data.data[:num_test_samples // 2].detach().clone().to(TORCH_DEVICE)
    sample_shape = pos_attack_models_test_data.shape[1:]
    neg_attack_models_test_data = []
    for _ in range(num_test_samples // 2):
        neg_sample = torch.normal(moments[0], moments[1])
        neg_sample = torch.where(neg_sample < 0, torch.zeros_like(neg_sample), torch.ones_like(neg_sample)).reshape(sample_shape)
        neg_attack_models_test_data.append(neg_sample.tolist())
    neg_attack_models_test_data = torch.tensor(neg_attack_models_test_data).to(TORCH_DEVICE)
    attack_models_test_data = torch.cat((pos_attack_models_test_data, neg_attack_models_test_data), dim=0)
    attack_models_test_targets = torch.cat((torch.ones(num_test_samples // 2, 1), torch.zeros(num_test_samples // 2, 1)), dim=0)
    attack_models_dset = GenericDataset(attack_models_test_data, attack_models_test_targets)

    print("Testing attack models...")
    membership_inference_attack.test_attack_models_dnn(attack_models_dset)
    print("Finished testing attack models.")

if __name__ == '__main__':
    main()
