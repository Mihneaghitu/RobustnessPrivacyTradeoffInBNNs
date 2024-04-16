import torch

from dataset_utils import get_marginal_distributions, load_mnist
from deterministic.membership_inference_dnn import MembershipInferenceAttack
from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE
from probabilistic.attack_types import AttackType
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import fgsm_predictive_distrib_attack, ibp_eval
from probabilistic.HMC.datasets import GenericDataset
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.membership_inference_bnn import \
    MembershipInferenceAttackBnn
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


def main():
    ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42
    torch.manual_seed(ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING)
    # membership_inference_dnn_experiment()
    membership_inference_bnn_experiment()
    return 0

def adversarial_robustness_experiment():
    print(f"Using device: {TORCH_DEVICE}")
    VANILLA_BNN = VanillaBnnLinear().to(TORCH_DEVICE)
    train_data, test_data = load_mnist("./")
    hyperparams_1 = HyperparamsHMC(num_epochs=70, num_burnin_epochs=15, step_size=0.01, lf_steps=600, criterion=torch.nn.CrossEntropyLoss(),
                                   batch_size=100, momentum_std=0.01, run_dp=False, grad_norm_bound=5, dp_sigma=0.1, alpha=0.6, eps=0.132)
    # 82.42%, 32.02%, 5.79% with IBP - 70, 15, 0.6
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

def membership_inference_dnn_experiment():
    # -------------- Hyperparams Values --------------
    BATCH_SIZE, NUM_EPOCHS, LR = 100, 20, 0.001
    train_data, _ = load_mnist()
    moments = get_marginal_distributions(train_data)
    TARGET_NETWORK = VanillaNetLinear().to(TORCH_DEVICE)
    TARGET_NETWORK.load_state_dict(torch.load('vanilla_network.pt'))
    membership_inference_attack = MembershipInferenceAttack(TARGET_NETWORK, TARGET_NETWORK.get_output_size(), moments)
    print("Training shadow models...")
    input_for_attack_models = membership_inference_attack.train_shadow_models(BATCH_SIZE, NUM_EPOCHS, LR)
    print("Finished training shadow models.")
    # ------------------------------------------------
    print("Training attack models...")
    membership_inference_attack.train_attack_models(input_for_attack_models, BATCH_SIZE, NUM_EPOCHS, LR)
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
    TARGET_NETWORK = VanillaBnnLinear().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=20, num_burnin_epochs=5, step_size=0.01, lf_steps=120, criterion=torch.nn.CrossEntropyLoss(),
                                 batch_size=100, momentum_std=0.01)
    hmc = HamiltonianMonteCarlo(TARGET_NETWORK, hyperparams)
    posterior_samples = hmc.train_mnist_vanilla(train_data)
    accuracy = hmc.test_hmc_with_average_logits(train_data, posterior_samples)
    print(f"Accuracy of BNN with average logits on training set: {accuracy} %")


    BATCH_SIZE, NUM_EPOCHS, LR = 100, 20, 0.001
    membership_inference_attack = MembershipInferenceAttackBnn(TARGET_NETWORK, TARGET_NETWORK.get_output_size(), moments, posterior_samples)
    print("Training shadow models...")
    input_for_attack_models = membership_inference_attack.train_shadow_models(BATCH_SIZE, NUM_EPOCHS, LR)
    print("Finished training shadow models.")
    # ------------------------------------------------
    print("Training attack models...")
    membership_inference_attack.train_attack_models(input_for_attack_models, BATCH_SIZE, NUM_EPOCHS, LR)
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
    #! Accuracy: 62.71%

if __name__ == '__main__':
    main()
