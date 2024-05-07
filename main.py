import torch

from common.attack_types import AttackType
from common.dataset_utils import get_marginal_distributions, load_mnist
from common.datasets import GenericDataset
from deterministic.attacks import \
    fgsm_test_set_attack as fgsm_test_set_attack_dnn
from deterministic.attacks import ibp_eval as ibp_eval_dnn
from deterministic.attacks import \
    pgd_test_set_attack as pgd_test_set_attack_dnn
from deterministic.hyperparams import Hyperparameters
from deterministic.membership_inference_dnn import MembershipInferenceAttack
from deterministic.pipeline import PipelineDnn
from deterministic.uncertainty import auroc as auroc_dnn
from deterministic.uncertainty import ece as ece_dnn
from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import (fgsm_predictive_distrib_attack,
                                       ibp_eval, pgd_predictive_distrib_attack)
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.membership_inference_bnn import \
    MembershipInferenceAttackBnn
from probabilistic.HMC.uncertainty import auroc, ece, ibp_auc_and_ece
from probabilistic.HMC.vanilla_bnn import VanillaBnnMnist


def main():
    #* Seed is set in globals.py
    # membership_inference_dnn_experiment()
    # membership_inference_bnn_experiment()
    # adv_dp_experiment()
    hmc_dp_experiment()
    # dnn_experiment(save_model=False)
    return 0

def adv_dp_experiment():
    print(f"Using device: {TORCH_DEVICE}")
    # vanilla_bnn = VanillaBnnMnist().to(TORCH_DEVICE)
    vanilla_bnn = VanillaBnnMnist().to(TORCH_DEVICE)
    train_data, test_data = load_mnist()
    hyperparams_1 = HyperparamsHMC(num_epochs=20, num_burnin_epochs=20, step_size=1e-4, warmup_step_size=0.2, lf_steps=120,
                                   batch_size=500, momentum_std=0.01, alpha=0.25, eps=0.1, decay_epoch_start=15, lr_decay_magnitude=0.1,
                                   eps_warmup_epochs=0, alpha_warmup_epochs=0, run_dp=False, grad_clip_bound=0.05,
                                   acceptance_clip_bound=0.01, tau_g=0.1, tau_l=0.1)

    hmc = AdvHamiltonianMonteCarlo(vanilla_bnn, hyperparams_1, attack_type=AttackType.IBP)

    samples = hmc.train_with_restarts(train_data, first_chain_from_trained=True)

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

    print('------------------- Uncertainty metrics -------------------')
    auc_val = auroc(hmc.net, test_data, samples)
    print(f'STD AUC: {auc_val}')
    ece_val = ece(hmc.net, test_data, samples)
    print(f'STD ECE: {ece_val}')
    fgsm_auc_val = auroc(hmc.net, adv_test_set_fgsm, samples)
    print(f'FGSM AUC: {fgsm_auc_val}')
    fgsm_ece_val = ece(hmc.net, adv_test_set_fgsm, samples)
    print(f'FGSM ECE: {fgsm_ece_val}')
    pgd_auc_val = auroc(hmc.net, adv_test_set_pgd, samples)
    print(f'PGD AUC: {pgd_auc_val}')
    pgd_ece_val = ece(hmc.net, adv_test_set_pgd, samples)
    print(f'PGD ECE: {pgd_ece_val}')
    ibp_auc_val, ibp_ece_val = ibp_auc_and_ece(hmc.net, test_data, samples, hmc.hps.eps)
    print(f'IBP AUC: {ibp_auc_val}')
    print(f'IBP ECE: {ibp_ece_val}')
    print('-----------------------------------------------------------')

def hmc_dp_experiment():
    train_data, test_data = load_mnist()
    print("Training Bayesian Neural Network using HMC...")
    target_network = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=120, num_burnin_epochs=40, step_size=0.005, lf_steps=100,
                                 batch_size=512, momentum_std=0.01, decay_epoch_start=40, lr_decay_magnitude=0.1,
                                 run_dp=True, grad_clip_bound=0.3, acceptance_clip_bound=0.3, tau_g=0.5, tau_l=0.5)
    hmc = HamiltonianMonteCarlo(target_network, hyperparams)
    posterior_samples = hmc.train_bnn(train_data)
    accuracy = hmc.test_hmc_with_average_logits(test_data, posterior_samples)
    print(f"Accuracy of BNN with average logits on training set: {accuracy} %")


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
    posterior_samples = hmc.train_bnn(train_data)
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

def dnn_experiment(test_only: bool = False, save_model: bool = False):
    net = VanillaNetLinear().to(TORCH_DEVICE)
    curr_dir = __file__.rsplit('/', 2)[0]
    hyperparams = Hyperparameters(num_epochs=120, lr=0.01, batch_size=60, lr_decay_magnitude=0.005, decay_epoch_start=35,
                                  alpha=1, eps=0.1, eps_warmup_itrs=10000, alpha_warmup_itrs=10000, warmup_itr_start=5000)

    pipeline = PipelineDnn(net, hyperparams, AttackType.IBP)
    if not test_only:
        train, test = load_mnist()
        pipeline.train_mnist_vanilla(train)
        if save_model:
            torch.save(pipeline.net.state_dict(), curr_dir + '/vanilla_network.pt')
    else:
        print("Testing only...")
        net = VanillaNetLinear()
        net.load_state_dict(torch.load(curr_dir + 'vanilla_network.pt'))

    hyperparams.eps = 0.05

    std_acc = pipeline.test_mnist_vanilla(test)
    print(f'Standard accuracy of the network on the 10000 test images: {std_acc}%')
    fgsm_test_set = fgsm_test_set_attack_dnn(net, hyperparams, test)
    fgsm_acc = pipeline.test_mnist_vanilla(fgsm_test_set)
    print(f'Accuracy of the network using FGSM adversarial test set: {fgsm_acc}%')
    pgd_test_set = pgd_test_set_attack_dnn(net, hyperparams, test)
    pgd_acc = pipeline.test_mnist_vanilla(pgd_test_set)
    print(f'Accuracy of the network using PGD adversarial test set: {pgd_acc}%')
    ibp_acc = ibp_eval_dnn(net, hyperparams, test)
    print(f'Accuracy of the network using IBP adversarial test set: {ibp_acc}%')
    auc_val = auroc_dnn(net, test)
    ece_val = ece_dnn(net, test)
    print(f'AUC: {auc_val}')
    print(f'ECE: {ece_val}')

if __name__ == '__main__':
    main()
