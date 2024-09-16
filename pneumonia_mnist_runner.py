
import os
import sys

import torch
import yaml
from torch.nn import BCEWithLogitsLoss

from common.attack_types import AttackType
from common.dataset_utils import load_fashion_mnist, load_pneumonia_mnist
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import ConvNetPneumoniaMnist
from experiments.experiment_utils import (compute_metrics_hmc,
                                          compute_metrics_sgd, resize,
                                          run_experiment_adv_hmc,
                                          run_experiment_hmc,
                                          run_experiment_sgd)
from globals import TORCH_DEVICE
from probabilistic.HMC.attacks import ibp_eval
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.uncertainty import auroc, ood_detection_auc_and_ece
from probabilistic.HMC.vanilla_bnn import ConvBnnPneumoniaMnist


def adv_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    # print the seed
    print(f"Using device: {TORCH_DEVICE}")
    conv_bnn = ConvBnnPneumoniaMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(
        num_epochs=80, num_burnin_epochs=25, step_size=0.04, batch_size=218, lr_decay_magnitude=0.5, lf_steps=24, num_chains=3,
        warmup_step_size=0.25, momentum_std=0.002, prior_std=5, alpha_warmup_epochs=16, eps_warmup_epochs=20, alpha=0.975, eps=0.01,
        run_dp=True, grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1, criterion=BCEWithLogitsLoss()
    )

    attack_type = AttackType.IBP
    model_name = "ADV-DP-HMC" if hyperparams.run_dp else "ADV-HMC"
    fname = "hmc_dp_pneumonia_mnist" if hyperparams.run_dp else "hmc_pneumonia_mnist"
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

    hmc, posterior_samples = run_experiment_adv_hmc(conv_bnn, TRAIN_DATA, hyperparams, attack_type, fname, init_from_trained=False)
    compute_metrics_hmc(hmc, TEST_DATA, posterior_samples, testing_eps=0.01, write_results=write_results,
                        model_name=model_name, dset_name="PNEUMONIA_MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)

def hmc_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    target_network = ConvBnnPneumoniaMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=150, num_burnin_epochs=10, step_size=0.065, lf_steps=24, criterion=BCEWithLogitsLoss(),
                                 num_chains=1, batch_size=218, momentum_std=0.01, decay_epoch_start=40, lr_decay_magnitude=0.5,
                                 run_dp=False, grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1)

    fname = f'''hmc{"_dp" if hyperparams.run_dp else ""}_pneumonia_mnist''' if save_model else None
    hmc, posterior_samples = run_experiment_hmc(target_network, TEST_DATA, hyperparams, save_dir_name=fname)
    model_name = "HMC-DP" if hyperparams.run_dp else "HMC"
    compute_metrics_hmc(hmc, TEST_DATA, posterior_samples, testing_eps=0.01, write_results=write_results,
                        model_name=model_name, dset_name="PNEUMONIA_MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)

def dnn_experiment(write_results: bool = False, save_model: bool = False, for_adv_comparison: bool = False, pre_train: bool = False):
    net = ConvNetPneumoniaMnist().to(TORCH_DEVICE)
    hyperparams = Hyperparameters(num_epochs=30, lr=0.07, batch_size=218, lr_decay_magnitude=0.001, decay_epoch_start=25,
                                  alpha=1, eps=0.01, eps_warmup_itrs=6000, alpha_warmup_itrs=10000, warmup_itr_start=3000,
                                  run_dp=False, grad_norm_bound=0.5, dp_sigma=0.1, criterion=BCEWithLogitsLoss())

    fname = "conv_net_pneumonia_mnist.pt" if save_model else None
    attack_type = AttackType.IBP
    pipeline = run_experiment_sgd(net, TRAIN_DATA, hyperparams, attack_type=attack_type, save_file_name=fname)
    if attack_type == AttackType.IBP and pre_train:
        name_for_bnn_init = "conv_net_pneumonia_mnist_ibp.pt" if not hyperparams.run_dp else "conv_net_pneumonia_mnist_ibp_dp.pt"
        torch.save(pipeline.net.state_dict(), "pre_trained/" + name_for_bnn_init)

    compute_metrics_sgd(pipeline, TEST_DATA, testing_eps=0.01, write_results=write_results, dset_name="PNEUMONIA_MNIST",
                        for_adv_comparison=for_adv_comparison)

def privacy_study():
    k = int(sys.argv[1])
    single_privacy_study(k)

def single_privacy_study(k):
    ood_data_test = load_fashion_mnist()[1]
    # Run the ablation study
    conv_net = ConvBnnPneumoniaMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(
        num_epochs=k, num_burnin_epochs=k//3, step_size=0.01, lf_steps=6, batch_size=872, num_chains=1, decay_epoch_start=50,
        lr_decay_magnitude=0.5, warmup_step_size=0.25, momentum_std=0.002, prior_mu=0.0, prior_std=5, alpha_warmup_epochs=k//3,
        eps_warmup_epochs=k//3, alpha=0.975, alpha_pre_trained=0.75, step_size_pre_trained=0.05, eps=0.075,
        run_dp=True, grad_clip_bound=0.1, acceptance_clip_bound=0.1, tau_g=1, tau_l=1, criterion=torch.nn.BCEWithLogitsLoss()
    )

    testing_eps = 0.01
    # run
    hmc, posterior_samples = run_experiment_adv_hmc(conv_net, TRAIN_DATA, hyperparams, AttackType.IBP, init_from_trained=False)
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
    results["PNEUMONIA_MNIST"].append({"num_epochs": k, "std_acc": std_acc, "ibp_acc": ibp_acc, "id_auroc": id_auroc, "ood_auroc": ood_auroc})
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(results, f)

def ablation_study():
    vary_eps = sys.argv[1] == "eps"
    vary_tau = sys.argv[1] == "tau"
    vary_prior = sys.argv[1] == "prior"
    vary_dset = sys.argv[1] == "dset"
    value = float(sys.argv[2])
    eps, tau_g, prior_std, sample_percentage, dim_ratio = 0.01, 0.1, 5, 1, 1 # defaults
    if vary_eps:
        eps = value
    elif vary_tau:
        tau_g = value
    elif vary_prior:
        prior_std = value
    elif vary_dset:
        sample_percentage = value
    else:
        dim_ratio = value
    single_ablation(eps=eps, tau_g=tau_g, prior_std=prior_std, sample_percentage=sample_percentage, dim_ratio=dim_ratio,
                    vary_eps=vary_eps, vary_tau=vary_tau, vary_prior=vary_prior, vary_dset=vary_dset)

def single_ablation(eps: float, tau_g: float, prior_std: float, sample_percentage: float, dim_ratio: int,
                    vary_eps: bool, vary_tau: bool, vary_prior: bool, vary_dset: bool):
    ood_data_test = load_fashion_mnist()[1]
    # Run the ablation study
    conv_net = ConvBnnPneumoniaMnist(dim_ratio=dim_ratio).to(TORCH_DEVICE)
    train_data_abl = TRAIN_DATA
    if vary_dset:
        train_data_abl = resize(TRAIN_DATA, sample_percentage)
    hyperparams = HyperparamsHMC(
        num_epochs=80, num_burnin_epochs=25, step_size=0.04, batch_size=218, lr_decay_magnitude=0.5, lf_steps=24, num_chains=3,
        warmup_step_size=0.25, momentum_std=0.002, prior_std=prior_std, alpha_warmup_epochs=16, eps_warmup_epochs=20, alpha=0.975, eps=eps,
        run_dp=True, grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=tau_g, tau_l=0.1, criterion=BCEWithLogitsLoss()
    )

    result_dict, testing_eps = None, 0.01
    # update hps
    hmc, posterior_samples = run_experiment_adv_hmc(conv_net, train_data_abl, hyperparams, AttackType.IBP)
    hyperparams.eps = testing_eps
    std_acc = hmc.test_hmc_with_average_logits(TEST_DATA, posterior_samples)
    ibp_acc = ibp_eval(conv_net, hyperparams, TEST_DATA, posterior_samples)
    id_auroc = auroc(conv_net, TEST_DATA, posterior_samples)
    ood_auroc = ood_detection_auc_and_ece(conv_net, TEST_DATA, ood_data_test, posterior_samples)[0]
    # write results to file
    fname = "experiments/ablation_pneumonia_paper.yaml"
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
        abl_type, varied_val = "model_size", dim_ratio
    if result_dict[abl_type] is None:
        result_dict[abl_type] = []
    result_dict[abl_type].append({"value": varied_val, "std_acc": std_acc, "ibp_acc": ibp_acc, "id_auroc": id_auroc, "ood_auroc": ood_auroc})
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(result_dict, f)

TRAIN_DATA, TEST_DATA = load_pneumonia_mnist()
# adv_dp_experiment(write_results=False, save_model=False, for_adv_comparison=False)
# hmc_dp_experiment(write_results=False, for_adv_comparison=False, save_model=False)
# dnn_experiment(save_model=False, write_results=False, for_adv_comparison=False)
ablation_study()
