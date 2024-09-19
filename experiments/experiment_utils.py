import copy
import os
import sys
from dataclasses import asdict
from typing import List, Tuple, Union

import torch
import yaml
from torch.utils.data import Dataset

sys.path.append("../")

from common.attack_types import AttackType
from common.dataset_utils import (load_fashion_mnist, load_mnist,
                                  load_pneumonia_mnist)
from common.datasets import GenericDataset
from deterministic.attacks import \
    fgsm_test_set_attack as fgsm_test_set_attack_dnn
from deterministic.attacks import ibp_eval as ibp_eval_dnn
from deterministic.attacks import \
    pgd_test_set_attack as pgd_test_set_attack_dnn
from deterministic.hyperparams import Hyperparameters
from deterministic.pipeline import PipelineDnn
from deterministic.uncertainty import auroc as auroc_dnn
from deterministic.uncertainty import ece as ece_dnn
from deterministic.uncertainty import \
    ood_detection_auc_and_ece as detect_ood_dnn
from deterministic.vanilla_net import (ConvNetPneumoniaMnist, VanillaNetLinear,
                                       VanillaNetMnist)
from globals import (MODEL_NAMES_ADV, MODEL_NAMES_ADV_DP, ROOT_FNAMES_ADV,
                     ROOT_FNAMES_ADV_DP, TORCH_DEVICE, AdvDpModel, AdvModel)
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import (fgsm_predictive_distrib_attack,
                                       ibp_eval, pgd_predictive_distrib_attack)
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.membership_inference_bnn import \
    MembershipInferenceAttackBnn
from probabilistic.HMC.uncertainty import auroc, ece, ood_detection_auc_and_ece
from probabilistic.HMC.vanilla_bnn import (ConvBnnPneumoniaMnist,
                                           VanillaBnnLinear, VanillaBnnMnist)


def run_experiment_hmc(net: VanillaBnnLinear, train_data: Dataset, hps: HyperparamsHMC,
                       save_dir_name: str = None) -> Tuple[HamiltonianMonteCarlo, List[torch.Tensor]]:
    hmc = HamiltonianMonteCarlo(net, hps)
    posterior_samples = hmc.train_with_restarts(train_data)
    if save_dir_name is not None:
        save_dir = __file__.rsplit('/', 1)[0] + "/posterior_samples/" + save_dir_name + "/"
        save_samples(posterior_samples, save_dir)

    return hmc, posterior_samples

def run_experiment_adv_hmc(net: VanillaBnnLinear, train_data: Dataset, hps: HyperparamsHMC, attack_type: AttackType, save_dir_name: str = None,
                           init_from_trained: bool = False) -> Tuple[AdvHamiltonianMonteCarlo, List[torch.Tensor]]:
    hmc = AdvHamiltonianMonteCarlo(net, hps, attack_type)
    posterior_samples = hmc.train_with_restarts(train_data, init_from_trained)
    if save_dir_name is not None:
        save_dir = __file__.rsplit('/', 1)[0] + "/posterior_samples/" + save_dir_name + "/"
        save_samples(posterior_samples, save_dir)

    return hmc, posterior_samples

def save_samples(posterior_samples: List[torch.Tensor], save_dir: str) -> None:
    # save_file_dir is the absolute path to the directory where the posterior samples will be saved
    os.makedirs(save_dir, exist_ok=True)
    for file in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, file))

    for i, sample in enumerate(posterior_samples):
        torch.save(sample, save_dir + f"sample_{i}.npy")

def load_samples(save_dir: str) -> List[torch.Tensor]:
    posterior_samples = []
    for file in os.listdir(save_dir):
        posterior_samples.append(torch.load(save_dir + file))

    return posterior_samples

def run_experiment_sgd(net: VanillaNetLinear, train_data: Dataset, hps: Hyperparameters, attack_type: AttackType, save_file_name: str = None) -> PipelineDnn:
    pipeline = PipelineDnn(net, hps, attack_type)
    pipeline.train_mnist_vanilla(train_data)
    if save_file_name is not None:
        save_dir = __file__.rsplit('/', 1)[0] + "/posterior_samples/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(pipeline.net.state_dict(), save_dir + save_file_name) # it comes as argument as ".pt" already

    return pipeline

def compute_metrics_hmc(hmc: Union[AdvHamiltonianMonteCarlo, HamiltonianMonteCarlo], test_data: Dataset, posterior_samples: torch.Tensor,
                        testing_eps: float = 0.1, write_results: bool = False, model_name: str = None, dset_name: str = None,
                        for_adv_comparison: bool = True) -> None:
    hmc.hps.eps = testing_eps

    print(f"Number of posterior samples: {len(posterior_samples)}")
    adv_test_set_fgsm = fgsm_predictive_distrib_attack(hmc.net, hmc.hps, test_data, posterior_samples)
    adv_test_set_pgd = pgd_predictive_distrib_attack(hmc.net, hmc.hps, test_data, posterior_samples)

    with torch.no_grad():
        # -------------------- Accuracy metrics --------------------
        print('------------------- Normal Accuracy -------------------')
        std_acc = hmc.test_hmc_with_average_logits(test_data, posterior_samples)
        print(f'Accuracy on standard test set: {std_acc} %')
        print('------------------- FGSM attack --------------------')
        fgsm_acc = hmc.test_hmc_with_average_logits(adv_test_set_fgsm, posterior_samples)
        print(f'Accuracy on FGSM adversarial test set: {fgsm_acc} %')
        print('------------------- PGD attack -------------------')
        pgd_acc = hmc.test_hmc_with_average_logits(adv_test_set_pgd, posterior_samples)
        print(f'Accuracy on PGD adversarial test set: {pgd_acc} %')
        ibp_acc = ibp_eval(hmc.net, hmc.hps, test_data, posterior_samples)
        print('------------------- IBP attack -------------------')
        print(f'Accuracy on IBP adversarial test set: {ibp_acc} %')

        # ------------------- Uncertainty metrics -------------------
        in_distrib_auroc = auroc(hmc.net, test_data, posterior_samples)
        in_distrib_ece = ece(hmc.net, test_data, posterior_samples)
        out_of_distrib_test_data = load_fashion_mnist()[1]
        ood_auroc, ood_ece = ood_detection_auc_and_ece(hmc.net, test_data, out_of_distrib_test_data, posterior_samples)

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

    test_func = pipeline.test_mnist_vanilla if pipeline.net.get_num_classes() > 1 else pipeline.test_binary_classification
    std_acc = test_func(test_data)
    fgsm_acc = test_func(fgsm_test_set)
    pgd_acc = test_func(pgd_test_set)
    ibp_acc = ibp_eval_dnn(pipeline.net, pipeline.hps, test_data)

    print('------------------- Normal Accuracy -------------------')
    print(f'Accuracy of SGD on standard test set: {std_acc} %')
    print('------------------- FGSM attack --------------------')
    print(f'Accuracy of SGD on FGSM adversarial test set: {fgsm_acc} %')
    print('------------------- PGD attack -------------------')
    print(f'Accuracy of SGD on PGD adversarial test set: {pgd_acc} %')
    print('------------------- IBP attack -------------------')
    print(f'Accuracy of SGD on IBP adversarial test set: {ibp_acc} %')

    # ------------------- Uncertainty metrics -------------------
    out_of_distrib_test = load_fashion_mnist()[1]
    in_distrib_auroc = auroc_dnn(pipeline.net, test_data)
    in_distrib_ece = ece_dnn(pipeline.net, test_data)
    ood_auroc, ood_ece = detect_ood_dnn(pipeline.net, test_data, out_of_distrib_test)

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

def load_results(dset_name: str, model_name: str, for_adv_comparison: bool = False) -> List[float]:
    results_file = __file__.rsplit('/', 1)[0] + "/"
    if for_adv_comparison:
        results_file += "results_adv.yaml"
    else:
        results_file += "results_all.yaml"

    with open(results_file, 'r', encoding="utf-8") as f:
        all_results = yaml.safe_load(f)
        return all_results[dset_name][model_name]

def load_ablations(dset_name: str, robustness: bool, privacy: bool) -> List[float]:
    results_file = __file__.rsplit('/', 1)[0] + "/"
    if dset_name == "MNIST":
        results_file += "ablation_mnist_paper.yaml"
    else:
        results_file += "ablation_pneumonia_paper.yaml"

    ablation_type = None
    if robustness:
        ablation_type = "rob"
    elif privacy:
        ablation_type = "priv"
    else:
        ablation_type = "unc"
    with open(results_file, 'r', encoding="utf-8") as f:
        all_results = yaml.safe_load(f)
        return all_results[ablation_type]

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

def test_hmc_from_file(test_set: Dataset, experiment_type: Union[AdvModel, AdvDpModel], dset_name="MNIST", testing_eps: float = 0.1):
    for_adv_comparison = isinstance(experiment_type, AdvModel)

    assert experiment_type != AdvModel.SGD or experiment_type != AdvDpModel.SGD, "Only HMC-BNN models are supported for this function"

    curr_dir = __file__.rsplit('/', 1)[0]
    config_file = curr_dir + ("/configs_adv.yaml" if for_adv_comparison else "/configs_all.yaml")
    posterior_samples_dir = curr_dir + "/posterior_samples/"
    model_name = None
    if for_adv_comparison:
        posterior_samples_dir += ROOT_FNAMES_ADV[experiment_type.value] + dset_name.lower() + "/"
        model_name = MODEL_NAMES_ADV[experiment_type.value]
    else:
        posterior_samples_dir += ROOT_FNAMES_ADV_DP[experiment_type.value] + dset_name.lower() + "/"
        model_name = MODEL_NAMES_ADV_DP[experiment_type.value]

    with open(config_file, 'r', encoding="utf-8") as f:
        all_configs = yaml.safe_load(f)
    hps_config = all_configs[dset_name][model_name]
    # This is saved as a string, but inside the class it's a module, so to avoid any weirdness and because it's optional anyways, we remove it
    del hps_config['criterion']
    hps, net = HyperparamsHMC(**hps_config), None
    posterior_samples = load_samples(posterior_samples_dir)
    if dset_name == "MNIST":
        # this can be a new object because the parameters are the posterior samples anyways
        net = VanillaBnnMnist().to(TORCH_DEVICE)
    elif dset_name == "PNEUMONIA_MNIST":
        net = ConvBnnPneumoniaMnist().to(TORCH_DEVICE)
        hps.criterion = torch.nn.BCEWithLogitsLoss()
    hmc = AdvHamiltonianMonteCarlo(net, hps)

    compute_metrics_hmc(hmc, test_set, posterior_samples, testing_eps=testing_eps, write_results=False, model_name=model_name,
                        dset_name=dset_name, for_adv_comparison=for_adv_comparison)

def test_dnn_from_file(test_set: Dataset, experiment_type: Union[AdvModel, AdvDpModel], dset_name="MNIST", testing_eps: float = 0.1):
    for_adv_comparison = isinstance(experiment_type, AdvModel)

    assert experiment_type == AdvModel.SGD or experiment_type == AdvDpModel.SGD, "Only DNN models are supported for this function"

    curr_dir = __file__.rsplit('/', 1)[0]
    config_file = curr_dir + ("/configs_adv.yaml" if for_adv_comparison else "/configs_all.yaml")
    net_file = curr_dir + "/posterior_samples/"
    model_name = "SGD"

    with open(config_file, 'r', encoding="utf-8") as f:
        all_configs = yaml.safe_load(f)
    hps_config = all_configs[dset_name][model_name]

    # This is saved as a string, but inside the class it's a module, so to avoid any weirdness and because it's optional anyways, we remove it
    del hps_config['criterion']
    hps, net = Hyperparameters(**hps_config), None
    if dset_name == "MNIST":
        # this can be a new object because the parameters are the posterior samples anyways
        net = VanillaNetMnist().to(TORCH_DEVICE)
        net_file += "vanilla_network.pt"
    elif dset_name == "PNEUMONIA_MNIST":
        net = ConvNetPneumoniaMnist().to(TORCH_DEVICE)
        hps.criterion = torch.nn.BCEWithLogitsLoss()
        net_file += "conv_net_pneumonia_mnist.pt"
    net.load_state_dict(torch.load(net_file))
    pipeline = PipelineDnn(net, hps)

    compute_metrics_sgd(pipeline, test_set, testing_eps=testing_eps, write_results=False, dset_name=dset_name, for_adv_comparison=for_adv_comparison)


def run_bnn_membership_inference_attack(train_data: Dataset, net: VanillaBnnLinear, moments: Tuple[torch.Tensor, torch.Tensor],
                                        posterior_samples: List[torch.Tensor]) -> None:
    batch_size, num_epochs, lr = 100, 20, 0.001
    membership_inference_attack = MembershipInferenceAttackBnn(net, net.get_num_classes(), moments, posterior_samples)
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
    membership_inference_attack.test_attack_models(attack_models_dset)
    print("Finished testing attack models.")


def resize(train_data: Dataset, sample_percentage: float):
    train_data_abl = copy.deepcopy(train_data)
    init_data, init_targets = copy.deepcopy(train_data.data.clone().detach()), copy.deepcopy(train_data.targets.clone().detach())
    new_data, new_targets = torch.tensor([]), torch.tensor([])
    for i in range(10): # number of classes
        indices = torch.where(init_targets == i)[0]
        num_samples_for_class = indices.shape[0]
        # choose %percentage of the indices randomly
        indices = torch.randperm(num_samples_for_class)[:int(sample_percentage * num_samples_for_class)]
        new_data = torch.cat((new_data, init_data[indices]))
        new_targets = torch.cat((new_targets, init_targets[indices]))
    train_data_abl.data = new_data.clone().detach()
    train_data_abl.targets = new_targets.clone().detach()

    return train_data_abl

def get_delta_dp_bound(eps_dp, num_chains, epochs, lf_steps, tau_l, tau_g) -> float:
    mu = (epochs / (2 * tau_l ** 2)) + 2 * (epochs * (lf_steps + 1) / (2 * tau_g ** 2))
    mu *= num_chains
    eps_dp, mu = torch.tensor(eps_dp), torch.tensor(mu)
    first_term = torch.erfc((eps_dp - mu) / (2 * torch.sqrt(mu)))
    second_term = torch.exp(eps_dp) * torch.erfc((eps_dp + mu) / (2 * torch.sqrt(mu)))
    delta = 0.5 * (first_term - second_term)

    return float(delta.item())

def get_delta_dp_bound_log(eps_dp, num_chains, epochs, lf_steps, tau_l, tau_g) -> float:
    mu = (epochs / (2 * tau_l ** 2)) + 2 * (epochs * (lf_steps + 1) / (2 * tau_g ** 2))
    mu *= num_chains
    print(f"mu: {mu}")
    eps_dp, mu = torch.tensor(eps_dp), torch.tensor(mu)
    first_term = torch.erfc((eps_dp - mu) / (2 * torch.sqrt(mu)))
    second_term_log = eps_dp + torch.log(torch.erfc((eps_dp + mu) / (2 * torch.sqrt(mu))))
    delta = 0.5 * (first_term - torch.exp(second_term_log))

    return float(delta.item())
