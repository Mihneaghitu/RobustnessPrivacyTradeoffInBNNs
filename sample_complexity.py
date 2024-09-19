import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import yaml
from torch.nn import BCEWithLogitsLoss

from common.attack_types import AttackType
from common.dataset_utils import (load_fashion_mnist, load_mnist,
                                  load_pneumonia_mnist, train_validation_split)
from common.datasets import Dataset
from experiments.experiment_utils import resize
from globals import TORCH_DEVICE
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import ibp_eval
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.uncertainty import ood_detection_auc_and_ece
from probabilistic.HMC.vanilla_bnn import (ConvBnnPneumoniaMnist,
                                           VanillaBnnLinear, VanillaBnnMnist)

DSET_RATIOS = (np.arange(5, 100, 5) / 100).tolist()
# BASE (BEST) HYPERPARAMETERS
BASE_MNIST_HYPERPARAMS = HyperparamsHMC(num_epochs=60, num_burnin_epochs=25, step_size=0.01, warmup_step_size=0.2, lf_steps=120,
    batch_size=500, num_chains=3, momentum_std=0.001, alpha=0.993, alpha_pre_trained=0.75, eps=0.075, step_size_pre_trained=0.001,
    decay_epoch_start=50, lr_decay_magnitude=0.5, eps_warmup_epochs=20, alpha_warmup_epochs=16, run_dp=True, grad_clip_bound=0.5,
    acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1, prior_std=15)
BASE_PNEUM_HYPERPARAMS = HyperparamsHMC(num_epochs=80, num_burnin_epochs=25, step_size=0.04, batch_size=218, lr_decay_magnitude=0.5,
    lf_steps=24, num_chains=3, warmup_step_size=0.25, momentum_std=0.002, prior_std=5, alpha_warmup_epochs=16, eps_warmup_epochs=20,
    alpha=0.975, eps=0.01, run_dp=True, grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1, criterion=BCEWithLogitsLoss())
# MODELS (NET ARCH)
MNIST_NET = VanillaBnnMnist().to(TORCH_DEVICE)
PNEUM_NET = ConvBnnPneumoniaMnist().to(TORCH_DEVICE)
# DATA (DATASETS)
MNIST_TRAIN, MNIST_TEST = load_mnist()
PNEUM_TRAIN, PNEUM_TEST = load_pneumonia_mnist()
OOD_TEST_SET = load_fashion_mnist()[1]
# RUNS
NUM_AVERAGING_RUNS = 5
# YAML FILE NAME
YAML_FILE = "experiments/sample_complexity.yaml"

def write_data(key: str, data: dict):
    result_dict = {}
    if not os.path.exists(YAML_FILE):
        open(YAML_FILE, "a", encoding="utf-8").close()
    else:
        with open(YAML_FILE, "r", encoding="utf-8") as f:
            result_dict = yaml.safe_load(f)

    if key not in result_dict:
        result_dict[key] = data

    with open(YAML_FILE, "w", encoding="utf-8") as f:
        yaml.dump(result_dict, f)

def run_sample_cplx_exp(varied_props: Dict[str, bool], thresholds: Dict[str, float], grids: Dict[str, List[Union[int, float]]], train_dset: Dataset,
                        test_dset: Dataset, hmc_runner: Union[HamiltonianMonteCarlo, AdvHamiltonianMonteCarlo]) -> tuple:
    order = ["epochs", "step_size", "num_chains", "lf_steps", "alpha"]
    grids_ls = [grids[prop] for prop in order if prop in varied_props] # equivalently, "if prop in grids"
    for dset_ratio in DSET_RATIOS:
        all_data = resize(train_dset, dset_ratio)
        train_data, validation_data = train_validation_split(all_data)
        hyperparams_for_avg_acc = {}
        for var_props in zip(*grids_ls):
            i = 0
            if varied_props["epochs"]:
                hmc_runner.hps.num_epochs = var_props[i]
                i += 1
            if varied_props["step_size"]:
                hmc_runner.hps.step_size = var_props[i]
                i += 1
            if varied_props["num_chains"]:
                hmc_runner.hps.num_chains = var_props[i]
                i += 1
            if varied_props["lf_steps"]:
                hmc_runner.hps.lf_steps = var_props[i]
                i += 1
            if varied_props["alpha"]:
                hmc_runner.hps.alpha = var_props[i]
                i += 1
            avg_vals_for_run = [0] * len(thresholds)
            for _ in range(NUM_AVERAGING_RUNS):
                posterior_samples = hmc_runner.train_with_restarts(train_data)
                k = 0
                if "acc" in thresholds:
                    acc = hmc_runner.test_hmc_with_average_logits(validation_data, posterior_samples)
                    avg_vals_for_run[k] += acc / NUM_AVERAGING_RUNS
                    k += 1
                if "unc" in thresholds:
                    ood_auroc = ood_detection_auc_and_ece(hmc_runner, validation_data, OOD_TEST_SET, posterior_samples)[0]
                    avg_vals_for_run[k] += ood_auroc / NUM_AVERAGING_RUNS
                    k += 1
                if "rob" in thresholds:
                    ibp_acc = ibp_eval(hmc_runner.model, hmc_runner.hps, validation_data, posterior_samples)
                    avg_vals_for_run[k] += ibp_acc / NUM_AVERAGING_RUNS
                    k += 1
            # so, keys of the thresholds are floats, tuple of floats are hashable, we can use them as keys of the dict
            hyperparams_for_avg_acc[tuple(avg_vals_for_run)] = tuple(var_props)
        best_hyperparams_tuple = hyperparams_for_avg_acc[max(hyperparams_for_avg_acc.keys())]
        i = 0
        if varied_props["epochs"]:
            hmc_runner.hps.num_epochs = best_hyperparams_tuple[i]
            i += 1
        if varied_props["step_size"]:
            hmc_runner.hps.step_size = best_hyperparams_tuple[i]
            i += 1
        if varied_props["num_chains"]:
            hmc_runner.hps.num_chains = best_hyperparams_tuple[i]
            i += 1
        if varied_props["lf_steps"]:
            hmc_runner.hps.lf_steps = best_hyperparams_tuple[i]
            i += 1
        if varied_props["alpha"]:
            hmc_runner.hps.alpha = best_hyperparams_tuple[i]
            i += 1
        best_posterior_samples = hmc_runner.train_with_restarts(all_data)
        test_vals = {}
        if "acc" in thresholds:
            acc = hmc_runner.test_hmc_with_average_logits(test_dset, best_posterior_samples)
            test_vals["acc"] = acc
        if "unc" in thresholds:
            ood_auroc = ood_detection_auc_and_ece(hmc_runner, test_dset, OOD_TEST_SET, best_posterior_samples)[0]
            test_vals["unc"] = ood_auroc
        if "rob" in thresholds:
            ibp_acc = ibp_eval(hmc_runner.model, hmc_runner.hps, test_dset, best_posterior_samples)
            test_vals["rob"] = ibp_acc
        num_better = 0
        for key in thresholds:
            if test_vals[key] >= thresholds[key]:
                num_better += 1
        if num_better == len(thresholds):
            hyperparams_dict = {}
            i = 0
            if varied_props["epochs"]:
                hyperparams_dict["epochs"] = best_hyperparams_tuple[i]
                i += 1
            if varied_props["step_size"]:
                hyperparams_dict["step_size"] = best_hyperparams_tuple[i]
                i += 1
            if varied_props["num_chains"]:
                hyperparams_dict["num_chains"] = best_hyperparams_tuple[i]
                i += 1
            if varied_props["lf_steps"]:
                hyperparams_dict["lf_steps"] = best_hyperparams_tuple[i]
                i += 1
            if varied_props["alpha"]:
                hyperparams_dict["alpha"] = best_hyperparams_tuple[i]
                i += 1
            hyperparams_dict["complexity"] = dset_ratio
            res_key = ""
            i = 0
            for key in thresholds:
                if i > 0:
                    res_key += "and"
                res_key += key
                i += 1
            write_data(res_key, hyperparams_dict)
            return tuple([dset_ratio] + list(best_hyperparams_tuple))


def get_acc_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset,
                              test_dset: Dataset, threshold: float) -> Tuple[float, int, float, int]:
    epochs_grid = [40, 50, 60]
    step_size_grid = [0.005, 0.01, 0.02]
    lf_steps_grid = [10, 20, 30]
    if model is ConvBnnPneumoniaMnist:
        lf_steps_grid = [12, 15, 18]

    hmc = HamiltonianMonteCarlo(model, hyperparams)
    hmc.hps.run_dp = False

    varied_props = {"epochs": True, "step_size": True, "lf_steps": True}
    grids = {"epochs": epochs_grid, "step_size": step_size_grid, "lf_steps": lf_steps_grid}
    thresholds = {"acc": threshold}

    dset_ratio, epoch_b, step_size_b, lf_steps_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, hmc)
    return dset_ratio, epoch_b, step_size_b, lf_steps_b

def get_unc_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset,
                              test_dset: Dataset, threshold: float) -> Tuple[float, int, float, int, int]:
    epochs_grid = [40, 50, 60, 70, 80]
    step_size_grid = [0.0125, 0.025, 0.0375, 0.05]
    num_chains_grid = [2, 3, 4]
    # base for mnist
    lf_steps_grid = [30, 40, 50, 60]
    if model is ConvBnnPneumoniaMnist:
        lf_steps_grid = [12, 15, 18, 21]

    hmc = HamiltonianMonteCarlo(model, hyperparams)
    hmc.hps.run_dp = False
    varied_props = {"epochs": True, "step_size": True, "num_chains": True, "lf_steps": True}
    grids = {"epochs": epochs_grid, "step_size": step_size_grid, "num_chains": num_chains_grid, "lf_steps": lf_steps_grid}
    thresholds = {"unc": threshold}

    dset_ratio, epoch_b, step_size_b, num_chains_b, lf_steps_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, hmc)
    return dset_ratio, epoch_b, step_size_b, num_chains_b, lf_steps_b

def get_rob_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset,
                              test_dset: Dataset, threshold: float) -> Tuple[float, int, float, int, float]:
    epochs_grid = [30, 40, 50]
    step_size_grid = [0.0125, 0.025, 0.0375, 0.05]
    # base for mnist
    alpha_grid = [0.75, 0.85, 0.95]
    lf_steps_grid = [30, 40, 50, 60]
    if model is ConvBnnPneumoniaMnist:
        lf_steps_grid = [12, 15, 18, 21]

    adv_hmc = AdvHamiltonianMonteCarlo(model, hyperparams, attack_type=AttackType.IBP)
    adv_hmc.hps.run_dp = False
    varied_props = {"epochs": True, "step_size": True, "lf_steps": True, "alpha": True}
    grids = {"epochs": epochs_grid, "step_size": step_size_grid, "lf_steps": lf_steps_grid, "alpha": alpha_grid}
    thresholds = {"rob": threshold}

    dset_ratio, epoch_b, step_size_b, lf_steps_b, alpha_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, adv_hmc)
    return dset_ratio, epoch_b, step_size_b, lf_steps_b, alpha_b

#! Technically, there is no threshold for privacy. It suffices to keep the same number of steps, epochs and chains as the base model
def get_priv_and_rob_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset, test_dset: Dataset,
                                       threshold_rob: float) -> Tuple[float, float, int]:
    step_size_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
    alpha_grid = [0.9, 0.925, 0.95, 0.975, 0.99]

    adv_hmc = AdvHamiltonianMonteCarlo(model, hyperparams, attack_type=AttackType.IBP)
    adv_hmc.hps.run_dp = True
    varied_props = {"step_size": True, "alpha": True}
    grids = {"step_size": step_size_grid, "alpha": alpha_grid}
    thresholds = {"rob": threshold_rob}

    dset_ratio, step_size_b, alpha_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, adv_hmc)
    return dset_ratio, step_size_b, alpha_b

def get_priv_and_unc_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset, test_dset: Dataset,
                                       threshold_unc: float) -> Tuple[float, float]:
    step_size_grid = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    adv_hmc = AdvHamiltonianMonteCarlo(model, hyperparams, attack_type=AttackType.IBP)
    adv_hmc.hps.run_dp = True
    varied_props = {"step_size": True}
    grids = {"step_size": step_size_grid}
    thresholds = {"unc": threshold_unc}

    dset_ratio, step_size_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, adv_hmc)
    return dset_ratio, step_size_b

def get_priv_and_acc_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset, test_dset: Dataset,
                                       threshold_acc: float) -> Tuple[float, int, float, int, int]:
    step_size_grid = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

    adv_hmc = AdvHamiltonianMonteCarlo(model, hyperparams, attack_type=AttackType.IBP)
    adv_hmc.hps.run_dp = True
    varied_props = {"step_size": True}
    grids = {"step_size": step_size_grid}
    thresholds = {"acc": threshold_acc}

    dset_ratio, step_size_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, adv_hmc)
    return dset_ratio, step_size_b

def get_acc_and_rob_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset, test_dset: Dataset,
                                       threshold_acc: float, threshold_rob: float) -> Tuple[float, int, float, int, int]:
    epochs_grid = [40, 50, 60]
    step_size_grid = [0.005, 0.01, 0.02]
    lf_steps_grid = [10, 20, 30]
    if model is ConvBnnPneumoniaMnist:
        lf_steps_grid = [12, 15, 18]
    alpha_grid = [0.75, 0.85, 0.95]

    hmc = AdvHamiltonianMonteCarlo(model, hyperparams, attack_type=AttackType.IBP)
    hmc.hps.run_dp = False

    varied_props = {"epochs": True, "step_size": True, "lf_steps": True, "alpha": True}
    grids = {"epochs": epochs_grid, "step_size": step_size_grid, "lf_steps": lf_steps_grid, "alpha": alpha_grid}
    thresholds = {"acc": threshold_acc, "rob": threshold_rob}

    dset_ratio, epoch_b, step_size_b, lf_steps_b, alpha_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, hmc)
    return dset_ratio, epoch_b, step_size_b, lf_steps_b, alpha_b

def get_acc_and_unc_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset, test_dset: Dataset,
                                       threshold_acc: float, threshold_unc: float) -> Tuple[float, int, float, int, int]:
    epochs_grid = [40, 50, 60, 70, 80]
    step_size_grid = [0.0125, 0.025, 0.0375, 0.05]
    num_chains_grid = [2, 3, 4]
    # base for mnist
    lf_steps_grid = [30, 40, 50, 60]
    if model is ConvBnnPneumoniaMnist:
        lf_steps_grid = [12, 15, 18, 21]

    hmc = HamiltonianMonteCarlo(model, hyperparams)
    hmc.hps.run_dp = False
    varied_props = {"epochs": True, "step_size": True, "num_chains": True, "lf_steps": True}
    grids = {"epochs": epochs_grid, "step_size": step_size_grid, "num_chains": num_chains_grid, "lf_steps": lf_steps_grid}
    thresholds = {"acc": threshold_acc, "unc": threshold_unc}

    dset_ratio, epoch_b, step_size_b, num_chains_b, lf_steps_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, hmc)
    return dset_ratio, epoch_b, step_size_b, num_chains_b, lf_steps_b

def get_unc_and_rob_sample_complexity(hyperparams: HyperparamsHMC, model: VanillaBnnLinear, train_dset: Dataset, test_dset: Dataset,
                                       threshold_unc: float, threshold_rob: float) -> Tuple[float, int, float, int, int]:
    epochs_grid = [40, 50, 60, 70, 80]
    step_size_grid = [0.0125, 0.025, 0.0375, 0.05]
    num_chains_grid = [2, 3, 4]
    # base for mnist
    lf_steps_grid = [30, 40, 50, 60]
    if model is ConvBnnPneumoniaMnist:
        lf_steps_grid = [12, 15, 18, 21]
    alpha_grid = [0.75, 0.85, 0.95]

    hmc = AdvHamiltonianMonteCarlo(model, hyperparams, attack_type=AttackType.IBP)
    hmc.hps.run_dp = False
    varied_props = {"epochs": True, "step_size": True, "num_chains": True, "lf_steps": True, "alpha": True}
    grids = {"epochs": epochs_grid, "step_size": step_size_grid, "num_chains": num_chains_grid, "lf_steps": lf_steps_grid, "alpha": alpha_grid}
    thresholds = {"unc": threshold_unc, "rob": threshold_rob}


    dset_ratio, epoch_b, step_size_b, num_chains_b, lf_steps_b, alpha_b = run_sample_cplx_exp(varied_props, thresholds, grids, train_dset, test_dset, hmc)
    return dset_ratio, epoch_b, step_size_b, num_chains_b, lf_steps_b, alpha_b
