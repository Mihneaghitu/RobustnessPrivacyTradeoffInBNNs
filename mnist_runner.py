import os
import sys

import torch
import yaml

from common.attack_types import AttackType
from common.dataset_utils import (get_marginal_distributions,
                                  load_fashion_mnist, load_mnist)
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import VanillaNetMnist
from experiments.experiment_utils import (compute_metrics_hmc,
                                          compute_metrics_sgd, load_samples,
                                          resize,
                                          run_bnn_membership_inference_attack,
                                          run_experiment_adv_hmc,
                                          run_experiment_hmc,
                                          run_experiment_sgd, save_samples)
from globals import TORCH_DEVICE
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import ibp_eval
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.uncertainty import auroc, ood_detection_auc_and_ece
from probabilistic.HMC.vanilla_bnn import VanillaBnnMnist


def adv_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    # print the seed
    print(f"Using device: {TORCH_DEVICE}")
    vanilla_bnn = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=60, num_burnin_epochs=25, step_size=0.01, warmup_step_size=0.2, lf_steps=120, batch_size=500,
                    num_chains=3, momentum_std=0.001, alpha=0.993, alpha_pre_trained=0.75, eps=0.075, step_size_pre_trained=0.001,
                    decay_epoch_start=50, lr_decay_magnitude=0.5, eps_warmup_epochs=20, alpha_warmup_epochs=16, run_dp=True, grad_clip_bound=0.5,
                    acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1, prior_std=15)

    attack_type = AttackType.IBP
    model_name = "ADV-DP-HMC" if hyperparams.run_dp else "ADV-HMC"
    fname = "hmc_dp_mnist" if hyperparams.run_dp else "hmc_mnist"
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

    hmc, posterior_samples = run_experiment_adv_hmc(vanilla_bnn, TRAIN_DATA, hyperparams, attack_type, fname, init_from_trained=True)
    compute_metrics_hmc(hmc, TEST_DATA, posterior_samples, testing_eps=0.05, write_results=write_results,
                        model_name=model_name, dset_name="MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)


def hmc_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    print("Training Bayesian Neural Network using HMC...")
    target_network = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=70, num_burnin_epochs=10, step_size=0.0165, lf_steps=120, num_chains=3,
                                 batch_size=500, momentum_std=0.01, decay_epoch_start=35, lr_decay_magnitude=0.5,
                                 run_dp=True, grad_clip_bound=0.4, acceptance_clip_bound=0.4, tau_g=0.2, tau_l=0.2)

    fname = f'''hmc{"_dp" if hyperparams.run_dp else ""}_mnist''' if save_model else None
    hmc, posterior_samples = run_experiment_hmc(target_network, TRAIN_DATA, hyperparams, save_dir_name=fname)
    model_name = "HMC-DP" if hyperparams.run_dp else "HMC"
    compute_metrics_hmc(hmc, TEST_DATA, posterior_samples, testing_eps=0.075, write_results=write_results,
                        model_name=model_name, dset_name="MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)

def dnn_experiment(write_results: bool = False, save_model: bool = False, for_adv_comparison: bool = False, pre_train: bool = False):
    net = VanillaNetMnist().to(TORCH_DEVICE)
    hyperparams = Hyperparameters(num_epochs=0, lr=0.1, batch_size=60, lr_decay_magnitude=0.5, decay_epoch_start=20,
                                  alpha=0.5, eps=0.1, eps_warmup_itrs=6000, alpha_warmup_itrs=10000, warmup_itr_start=3000,
                                  run_dp=True, grad_norm_bound=0.5, dp_sigma=0.1)

    fname = "vanilla_network.pt" if save_model else None
    attack_type = AttackType.IBP
    pipeline = run_experiment_sgd(net, TRAIN_DATA, hyperparams, attack_type=attack_type, save_file_name=fname)
    if attack_type == AttackType.IBP and pre_train:
        name_for_bnn_init = "vanilla_network_ibp" if not hyperparams.run_dp else "vanilla_network_ibp_dp"
        torch.save(pipeline.net.state_dict(), f"pre_trained/{name_for_bnn_init}.pt")

    compute_metrics_sgd(pipeline, TEST_DATA, testing_eps=0.075, write_results=write_results, dset_name="MNIST", for_adv_comparison=for_adv_comparison)

def privacy_study():
    k = int(sys.argv[1])
    single_privacy_study(k)

def single_privacy_study(k):
    ood_data_test = load_fashion_mnist()[1]
    # Run the ablation study
    conv_net = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(
        num_epochs=k, num_burnin_epochs=k//3, step_size=0.01, lf_steps=10, batch_size=6000, num_chains=1, decay_epoch_start=50,
        lr_decay_magnitude=0.5, warmup_step_size=0.25, momentum_std=0.0015, prior_mu=0.0, prior_std=10, alpha_warmup_epochs=k//3,
        eps_warmup_epochs=k//3, alpha=0.99, alpha_pre_trained=0.75, step_size_pre_trained=0.05, eps=0.075,
        run_dp=True, grad_clip_bound=0.1, acceptance_clip_bound=0.1, tau_g=2, tau_l=2
    )

    testing_eps = 0.075
    save_dir = __file__.rsplit('/', 1)[0] + "experiments/privacy_study/" + f"k{k}/"
    # run
    hmc, posterior_samples = run_experiment_adv_hmc(conv_net, TRAIN_DATA, hyperparams, AttackType.IBP, init_from_trained=True)
    save_samples(posterior_samples, save_dir)
    hyperparams.eps = testing_eps
    std_acc = hmc.test_hmc_with_average_logits(TEST_DATA, posterior_samples)
    ibp_acc = ibp_eval(conv_net, hyperparams, TEST_DATA, posterior_samples)
    id_auroc = auroc(conv_net, TEST_DATA, posterior_samples)
    ood_auroc = ood_detection_auc_and_ece(conv_net, TEST_DATA, ood_data_test, posterior_samples)[0]

    fname = "experiments/privacy_study.yaml"
    if not os.path.exists(fname):
        open(fname, "a", encoding="utf-8").close()
    with open(fname, "r", encoding="utf-8") as f:
        results = yaml.safe_load(f)
    if results is None:
        results = {"MNIST": [], "PNEUMONIA_MNIST": []}
    results["MNIST"].append({"num_epochs": k, "std_acc": std_acc, "ibp_acc": ibp_acc, "id_auroc": id_auroc, "ood_auroc": ood_auroc})
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(results, f)

def ablation_study():
    vary_eps = sys.argv[1] == "eps"
    vary_tau = sys.argv[1] == "tau"
    vary_prior = sys.argv[1] == "prior"
    vary_dset = sys.argv[1] == "dset"
    value = float(sys.argv[2])
    eps, tau_g, prior_std, sample_percentage, hidden_dim = 0.075, 0.05, 15, 1, 512 # defaults
    if vary_eps:
        eps = value
    elif vary_tau:
        tau_g = value
    elif vary_prior:
        prior_std = value
    elif vary_dset:
        sample_percentage = value
    else:
        hidden_dim = value
    single_ablation(eps=eps, tau_g=tau_g, prior_std=prior_std, sample_percentage=sample_percentage, hidden_dim=hidden_dim,
                    vary_eps=vary_eps, vary_tau=vary_tau, vary_prior=vary_prior, vary_dset=vary_dset)

def single_ablation(eps: float, tau_g: float, prior_std: float, sample_percentage: float, hidden_dim: int,
                    vary_eps: bool, vary_tau: bool, vary_prior: bool, vary_dset: bool):
    ood_data_test = load_fashion_mnist()[1]
    # Run the ablation study
    net = VanillaBnnMnist([hidden_dim]).to(TORCH_DEVICE)
    train_data_abl = TRAIN_DATA
    if vary_dset:
        # get the percentage of each class of the dataset (just for training)
        train_data_abl = resize(TRAIN_DATA, sample_percentage)
    hyperparams = HyperparamsHMC(
        num_epochs=60, num_burnin_epochs=25, step_size=0.01, lf_steps=120, batch_size=500, num_chains=3, decay_epoch_start=50,
        lr_decay_magnitude=0.5, warmup_step_size=0.2, momentum_std=0.001, prior_mu=0.0, prior_std=prior_std, alpha_warmup_epochs=16,
        eps_warmup_epochs=20, alpha=0.993, alpha_pre_trained=0.75, step_size_pre_trained=0.001, eps=eps, run_dp=True,
        grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=tau_g, tau_l=0.05
    )

    result_dict, testing_eps = None, 0.075
    # update hps
    hmc, posterior_samples = run_experiment_adv_hmc(net, train_data_abl, hyperparams, AttackType.IBP)
    hyperparams.eps = testing_eps
    std_acc = hmc.test_hmc_with_average_logits(TEST_DATA, posterior_samples)
    ibp_acc = ibp_eval(net, hyperparams, TEST_DATA, posterior_samples)
    id_auroc = auroc(net, TEST_DATA, posterior_samples)
    ood_auroc = ood_detection_auc_and_ece(net, TEST_DATA, ood_data_test, posterior_samples)[0]
    # write results to file
    fname = "experiments/ablation_mnist_paper.yaml"
    if not os.path.exists(fname):
        open(fname, "a", encoding="utf-8").close()
        result_dict = {"rob": [], "priv": [], "unc": [], "dset": [], "model_size": []}
    else:
        with open(fname, "r", encoding="utf-8") as f:
            result_dict = yaml.safe_load(f)

    abl_type, varied_val = None, None
    if vary_eps:
        abl_type, varied_val = "rob", eps
    elif vary_tau:
        abl_type, varied_val = "priv", tau_g
    elif vary_prior:
        abl_type, varied_val = "unc", prior_std
    elif vary_dset:
        abl_type, varied_val = "dset", sample_percentage
    else:
        abl_type, varied_val = "model_size", hidden_dim
    result_dict[abl_type].append({"value": varied_val, "std_acc": std_acc, "ibp_acc": ibp_acc, "id_auroc": id_auroc, "ood_auroc": ood_auroc})
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(result_dict, f)

def membership_inference_bnn_experiment(adv: bool = False, load_dir: bool = False, attack_type: AttackType = None):
    moments = get_marginal_distributions(TRAIN_DATA)
    target_network = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(
        num_epochs=60, num_burnin_epochs=25, step_size=0.01, warmup_step_size=0.2, lf_steps=120, batch_size=500, num_chains=3,
        momentum_std=0.001, alpha=0.993, alpha_pre_trained=0.75, eps=0.075, step_size_pre_trained=0.001, decay_epoch_start=50,
        lr_decay_magnitude=0.5, eps_warmup_epochs=20, alpha_warmup_epochs=16, run_dp=True, grad_clip_bound=0.5,
        acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1, prior_std=15
    )

    hmc, posterior_samples = None, None
    if adv:
        hmc = AdvHamiltonianMonteCarlo(target_network, hyperparams, attack_type)
    else:
        hmc = HamiltonianMonteCarlo(target_network, hyperparams)
    if load_dir is not None:
        savedir = "experiments/posterior_samples/" + load_dir
        posterior_samples = load_samples(savedir)
    else:
        hmc, posterior_samples = run_experiment_hmc(target_network, TRAIN_DATA, hyperparams)

    run_bnn_membership_inference_attack(TRAIN_DATA, target_network, moments, posterior_samples)

TRAIN_DATA, TEST_DATA = load_mnist()
# adv_dp_experiment(write_results=True, save_model=True, for_adv_comparison=False)
# hmc_dp_experiment(write_results=False, for_adv_comparison=False, save_model=False)
# dnn_experiment(save_model=False, write_results=False, for_adv_comparison=False)
# ablation_study()
# privacy_study()
# membership_inference_bnn_experiment(adv=True, load_dir="ibp_hmc_dp_mnist/", attack_type=AttackType.IBP)
ablation_study()
