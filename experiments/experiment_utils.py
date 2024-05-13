import os
import sys
from dataclasses import asdict
from typing import List, Tuple, Union

import torch
import yaml
from torch.utils.data import Dataset

sys.path.append("../")

from common.attack_types import AttackType
from common.dataset_utils import load_fashion_mnist
from deterministic.attacks import \
    fgsm_test_set_attack as fgsm_test_set_attack_dnn
from deterministic.attacks import ibp_eval as ibp_eval_dnn
from deterministic.attacks import \
    pgd_test_set_attack as pgd_test_set_attack_dnn
from deterministic.hyperparams import Hyperparameters
from deterministic.pipeline import PipelineDnn
from deterministic.uncertainty import auroc as auroc_dnn
from deterministic.uncertainty import ece as ece_dnn
from deterministic.vanilla_net import VanillaNetLinear
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import (fgsm_predictive_distrib_attack,
                                       ibp_eval, pgd_predictive_distrib_attack)
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.uncertainty import auroc, ece
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


def run_experiment_hmc(net: VanillaBnnLinear, train_data: Dataset, hps: HyperparamsHMC,
                       save_file_name: str = None) -> Tuple[HamiltonianMonteCarlo, List[torch.Tensor]]:
    hmc = HamiltonianMonteCarlo(net, hps)
    posterior_samples = hmc.train_bnn(train_data)
    if save_file_name is not None:
        save_dir = __file__.rsplit('/', 1)[0] + "/posterior_samples/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(posterior_samples, save_dir + save_file_name + ".npy")

    return hmc, posterior_samples

def run_experiment_adv_hmc(net: VanillaBnnLinear, train_data: Dataset, hps: HyperparamsHMC, attack_type: AttackType, save_file_name: str = None,
                           init_from_trained: bool = False) -> Tuple[AdvHamiltonianMonteCarlo, List[torch.Tensor]]:
    hmc = AdvHamiltonianMonteCarlo(net, hps, attack_type)
    posterior_samples = hmc.train_with_restarts(train_data, init_from_trained)
    if save_file_name is not None:
        save_dir = __file__.rsplit('/', 1)[0] + "/posterior_samples/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(posterior_samples, save_dir + save_file_name + ".npy")

    return hmc, posterior_samples

def run_experiment_sgd(net: VanillaNetLinear, train_data: Dataset, hps: Hyperparameters, attack_type: AttackType, save_file_name: str = None) -> PipelineDnn:
    pipeline = PipelineDnn(net, hps, attack_type)
    pipeline.train_mnist_vanilla(train_data)
    if save_file_name is not None:
        save_dir = __file__.rsplit('/', 1)[0] + "/posterior_samples/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(pipeline.net.state_dict(), save_dir + save_file_name + ".npy")

    return pipeline

def compute_metrics_hmc(hmc: Union[AdvHamiltonianMonteCarlo, HamiltonianMonteCarlo], test_data: Dataset, posterior_samples: torch.Tensor,
                        testing_eps: float = 0.1, write_results: bool = False, model_name: str = None, dset_name: str = None,
                        for_adv_comparison: bool = True) -> None:
    hmc.hps.eps = testing_eps

    # -------------------- Accuracy metrics --------------------
    adv_test_set_pgd = pgd_predictive_distrib_attack(hmc.net, hmc.hps, test_data, posterior_samples)
    adv_test_set_fgsm = fgsm_predictive_distrib_attack(hmc.net, hmc.hps, test_data, posterior_samples)

    print(f"Number of posterior samples: {len(posterior_samples)}")
    print('------------------- Normal Accuracy -------------------')
    std_acc = hmc.test_hmc_with_average_logits(test_data, posterior_samples)
    print(f'Accuracy of ADV-HMC-DP with average logit on standard test set: {std_acc} %')
    print('------------------- FGSM attack --------------------')
    fgsm_acc = hmc.test_hmc_with_average_logits(adv_test_set_fgsm, posterior_samples)
    print(f'Accuracy of ADV-HMC-DP with average logit on FGSM adversarial test set: {fgsm_acc} %')
    print('------------------- PGD attack -------------------')
    pgd_acc = hmc.test_hmc_with_average_logits(adv_test_set_pgd, posterior_samples)
    print(f'Accuracy of ADV-HMC-DP with average logit on PGD adversarial test set: {pgd_acc} %')
    print('------------------- IBP attack -------------------')
    with torch.no_grad():
        ibp_acc = ibp_eval(hmc.net, hmc.hps, test_data, posterior_samples)
    print(f'Accuracy of ADV-HMC-DP with IBP on standard test set: {ibp_acc} %')

    # ------------------- Uncertainty metrics -------------------
    in_distrib_auroc = auroc(hmc.net, test_data, posterior_samples)
    in_distrib_ece = ece(hmc.net, test_data, posterior_samples)
    out_of_distrib_test_data = load_fashion_mnist()[1]
    ood_auroc = auroc(hmc.net, out_of_distrib_test_data, posterior_samples)
    ood_ece = ece(hmc.net, out_of_distrib_test_data, posterior_samples)

    print('------------------- Uncertainty metrics -------------------')
    print(f'STD AUC: {in_distrib_auroc}')
    print(f'STD ECE: {in_distrib_ece}')
    print(f'OUT-OF-DISTRIBUTION AUC: {ood_auroc}')
    print(f'OUT-OF-DISTRIBUTION ECE: {ood_ece}')
    print('-----------------------------------------------------------')

    if write_results:
        std_acc = round(float(std_acc) / 100, 4)
        fgsm_acc = round(float(fgsm_acc) / 100, 4)
        pgd_acc = round(float(pgd_acc) / 100, 4)
        ibp_acc = round(float(ibp_acc) / 100, 4)
        results = {"STD_ACC": std_acc, "FGSM_ACC": fgsm_acc, "PGD_ACC": pgd_acc, "IBP_ACC": ibp_acc,
                   "IN_DISTRIB_AUROC": in_distrib_auroc, "IN_DISTRIB_ECE": in_distrib_ece, "OOD_AUROC": ood_auroc, "OOD_ECE": ood_ece}
        save_results(dset_name, model_name, results, for_adv_comparison)
        hmc.hps = hmc.init_hps # Reset the hyperparameters to the initial values
        save_config(hmc.hps, model_name, dset_name, for_adv_comparison)

def compute_metrics_sgd(pipeline: PipelineDnn, test_data: Dataset, testing_eps: float = 0.1,
                        write_results: bool = False, dset_name: str = None, for_adv_comparison: bool = True) -> None:
    pipeline.hps.eps = testing_eps

    # -------------------- Accuracy metrics --------------------
    fgsm_test_set = fgsm_test_set_attack_dnn(pipeline.net, pipeline.hps, test_data)
    pgd_test_set = pgd_test_set_attack_dnn(pipeline.net, pipeline.hps, test_data)

    std_acc = pipeline.test_mnist_vanilla(test_data)
    fgsm_acc = pipeline.test_mnist_vanilla(fgsm_test_set)
    pgd_acc = pipeline.test_mnist_vanilla(pgd_test_set)
    ibp_acc = ibp_eval_dnn(pipeline.net, pipeline.hps, test_data)

    print('------------------- Normal Accuracy -------------------')
    print(f'Accuracy of ADV-HMC-DP with average logit on standard test set: {std_acc} %')
    print('------------------- FGSM attack --------------------')
    print(f'Accuracy of ADV-HMC-DP with average logit on FGSM adversarial test set: {fgsm_acc} %')
    print('------------------- PGD attack -------------------')
    print(f'Accuracy of ADV-HMC-DP with average logit on PGD adversarial test set: {pgd_acc} %')
    print('------------------- IBP attack -------------------')
    print(f'Accuracy of ADV-HMC-DP with IBP on standard test set: {ibp_acc} %')

    # ------------------- Uncertainty metrics -------------------
    out_of_distrib_test = load_fashion_mnist()[1]
    in_distrib_auroc = auroc_dnn(pipeline.net, test_data)
    in_distrib_ece = ece_dnn(pipeline.net, test_data)
    ood_auroc = auroc_dnn(pipeline.net, out_of_distrib_test)
    ood_ece = ece_dnn(pipeline.net, out_of_distrib_test)

    print('------------------- Uncertainty metrics -------------------')
    print(f'STD AUC: {in_distrib_auroc}')
    print(f'STD ECE: {in_distrib_ece}')
    print(f'OUT-OF-DISTRIBUTION AUC: {ood_auroc}')
    print(f'OUT-OF-DISTRIBUTION ECE: {ood_ece}')
    print('-----------------------------------------------------------')

    if write_results:
        std_acc = round(float(std_acc) / 100, 4)
        fgsm_acc = round(float(fgsm_acc) / 100, 4)
        pgd_acc = round(float(pgd_acc) / 100, 4)
        ibp_acc = round(float(ibp_acc) / 100, 4)
        results = {"STD_ACC": std_acc, "FGSM_ACC": fgsm_acc, "PGD_ACC": pgd_acc, "IBP_ACC": ibp_acc,
                   "IN_DISTRIB_AUROC": in_distrib_auroc, "IN_DISTRIB_ECE": in_distrib_ece, "OOD_AUROC": ood_auroc, "OOD_ECE": ood_ece}
        save_results(dset_name, "SGD", results, for_adv_comparison)
        pipeline.hps = pipeline.init_hps # Reset the hyperparameters to the initial values
        save_config(pipeline.hps, "SGD", dset_name, for_adv_comparison)

def save_config(hps: Union[Hyperparameters, HyperparamsHMC], model_name: str, dset_name: str, for_adv_comparison: bool = True) -> None:
    config_file = __file__.rsplit('/', 1)[0] + "/"
    if for_adv_comparison:
        config_file += "configs_adv.yaml"
    else:
        config_file += "configs_all.yaml"

    current_config = asdict(hps)
    current_config["criterion"] = current_config["criterion"].__class__.__name__
    all_configs = {}
    with open(config_file, 'r', encoding="utf-8") as f:
        all_configs = yaml.safe_load(f)
        all_configs[dset_name][model_name] = current_config

    with open(config_file, 'w', encoding="utf-8") as f:
        yaml.dump(all_configs, f, sort_keys=False)

def save_results(dset_name: str, model_name: str, results: dict, for_adv_comparison: bool = True) -> None:
    results_file = __file__.rsplit('/', 1)[0] + "/"
    if for_adv_comparison:
        results_file += "results_adv.yaml"
    else:
        results_file += "results_all.yaml"

    with open(results_file, 'r', encoding="utf-8") as f:
        all_results = yaml.safe_load(f)
        for changed_metric, new_vals in results.items():
            all_results[dset_name][model_name][changed_metric] = new_vals

    with open(results_file, 'w', encoding="utf-8") as f:
        yaml.dump(all_results, f, sort_keys=False)

#* It is fine to just pass the AdvHamiltonianMonteCarlo object with the net untrained,
#* because the net params are taken from the posterior samples in the file anyways
def test_ibp_acc_from_file(hmc: AdvHamiltonianMonteCarlo, test_data: Dataset, fname: str, epsilons: List[float]) -> List[float]:
    posterior_samples = torch.load(fname)
    accs = []
    for testing_eps in epsilons:
        hmc.hps.eps = testing_eps
        with torch.no_grad():
            curr_acc = ibp_eval(hmc.net, hmc.hps, test_data, posterior_samples)
            accs.append(round(float(curr_acc) / 100, 4))

    return accs
