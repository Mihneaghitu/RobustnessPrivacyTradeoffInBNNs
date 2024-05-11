import torch
import yaml

from common.attack_types import AttackType
from common.dataset_utils import get_marginal_distributions, load_mnist
from common.datasets import GenericDataset
from deterministic.hyperparams import Hyperparameters
from deterministic.membership_inference_dnn import MembershipInferenceAttack
from deterministic.pipeline import PipelineDnn
from deterministic.vanilla_net import VanillaNetLinear
from experiments.experiment_utils import (compute_metrics_hmc,
                                          compute_metrics_sgd,
                                          run_experiment_adv_hmc,
                                          run_experiment_hmc,
                                          run_experiment_sgd)
from globals import TORCH_DEVICE
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.membership_inference_bnn import \
    MembershipInferenceAttackBnn
from probabilistic.HMC.vanilla_bnn import VanillaBnnMnist


def main():
    #* Seed is set in globals.py
    # membership_inference_dnn_experiment()
    # membership_inference_bnn_experiment()
    adv_dp_experiment(write_results=True, save_model=True)
    # hmc_dp_experiment(write_results=True)
    # dnn_experiment(save_model=False, write_results=False)
    return 0

def adv_dp_experiment(write_results: bool = False, save_model: bool = False):
    print(f"Using device: {TORCH_DEVICE}")
    vanilla_bnn = VanillaBnnMnist().to(TORCH_DEVICE)
    train_data, test_data = load_mnist()
    hyperparams = HyperparamsHMC(num_epochs=70, num_burnin_epochs=0, step_size=4e-3, warmup_step_size=0.75, lf_steps=120,
                                 batch_size=500, momentum_std=0.002, alpha=0.4, eps=0.1, decay_epoch_start=20,
                                 lr_decay_magnitude=0.1, eps_warmup_epochs=0, alpha_warmup_epochs=0, run_dp=False,
                                 grad_clip_bound=0.25, acceptance_clip_bound=0.25, tau_g=0.5, tau_l=0.5)

    attack_type = AttackType.IBP
    model_name = "ADV-DP-HMC" if hyperparams.run_dp else "ADV-HMC"
    fname = "hmc_mnist_dp" if hyperparams.run_dp else "hmc_mnist"
    if attack_type == AttackType.FGSM:
        model_name += " (FGSM)"
        fname = "fgsm_" + fname
    elif attack_type == AttackType.PGD:
        model_name += " (PGD)"
        fname = "pgd_" + fname
    else:
        model_name += " (IBP)"
        fname = "ibp_" + fname
    fname = fname if save_model else None

    hmc, posterior_samples = run_experiment_adv_hmc(vanilla_bnn, train_data, hyperparams, attack_type, fname, init_from_trained=True)
    compute_metrics_hmc(hmc ,test_data, posterior_samples, testing_eps=0.075, write_results=write_results,
                        model_name=model_name, dset_name="MNIST")


def hmc_dp_experiment(write_results: bool = False, save_model: bool = False):
    train_data, test_data = load_mnist()
    print("Training Bayesian Neural Network using HMC...")
    target_network = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=100, num_burnin_epochs=25, step_size=0.02, lf_steps=100,
                                 batch_size=512, momentum_std=0.01, decay_epoch_start=25, lr_decay_magnitude=0.05,
                                 run_dp=False, grad_clip_bound=2, acceptance_clip_bound=0.5, tau_g=0.5, tau_l=0.5)

    fname = f'''hmc_mnist{"_dp" if hyperparams.run_dp else ""}.pt''' if save_model else None
    hmc, posterior_samples = run_experiment_hmc(target_network, train_data, hyperparams, save_file_name=fname)
    model_name = "HMC-DP" if hyperparams.run_dp else "HMC"
    compute_metrics_hmc(hmc ,test_data, posterior_samples, testing_eps=0.075, write_results=write_results,
                        model_name=model_name, dset_name="MNIST")

def dnn_experiment(save_model: bool = False, write_results: bool = False):
    train, test = load_mnist()
    net = VanillaNetLinear().to(TORCH_DEVICE)
    hyperparams = Hyperparameters(num_epochs=12, lr=0.01, batch_size=60, lr_decay_magnitude=0.1, decay_epoch_start=15,
                                  alpha=0.5, eps=0.1, eps_warmup_itrs=4000, alpha_warmup_itrs=4000, warmup_itr_start=3000)

    fname = "vanilla_network.pt" if save_model else None
    pipeline = run_experiment_sgd(net, train, hyperparams, AttackType.IBP, save_file_name=fname)
    torch.save(pipeline.net.state_dict(), "vanilla_network.pt")

    compute_metrics_sgd(pipeline, test, testing_eps=0.075, write_results=write_results, dset_name="MNIST")

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

if __name__ == '__main__':
    main()
