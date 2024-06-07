import torch
import yaml

from common.attack_types import AttackType
from common.dataset_utils import (get_marginal_distributions,
                                  load_fashion_mnist, load_mnist)
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import VanillaNetMnist
from experiments.experiment_utils import (compute_metrics_hmc,
                                          compute_metrics_sgd, load_samples,
                                          run_bnn_membership_inference_attack,
                                          run_experiment_adv_hmc,
                                          run_experiment_hmc,
                                          run_experiment_sgd, save_samples)
from globals import TORCH_DEVICE
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import ibp_eval
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
    ood_data_test = load_fashion_mnist()[1]
    # Run the ablation study
    conv_net = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(
        num_epochs=1, num_burnin_epochs=0, step_size=0.01, lf_steps=10, batch_size=6000, num_chains=3, decay_epoch_start=50,
        lr_decay_magnitude=0.5, warmup_step_size=0.2, momentum_std=0.001, prior_mu=0.0, prior_std=15, alpha_warmup_epochs=0,
        eps_warmup_epochs=0, alpha=0.993, alpha_pre_trained=0.75, step_size_pre_trained=0.001, eps=0.075,
        run_dp=True, grad_clip_bound=0.05, acceptance_clip_bound=0.05, tau_g=4, tau_l=4
    )

    testing_eps = 0.075
    num_epochs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    dirnames = []
    result_dict = {"num_epochs": []}
    save_dir = __file__.rsplit('/', 1)[0] + "experiments/privacy_study/"
    for k in num_epochs:
        dirnames.append(save_dir + f"k{k}/")
    for k, dirname in zip(num_epochs, dirnames):
        # update hps
        hyperparams.num_epochs = k
        if k >= 3:
            hyperparams.num_burnin_epochs = hyperparams.eps_warmup_epochs = hyperparams.alpha_warmup_epochs = 2 * k // 3
        hmc, posterior_samples = run_experiment_adv_hmc(conv_net, TRAIN_DATA, hyperparams, AttackType.IBP, init_from_trained=True)
        save_samples(posterior_samples, dirname)
        hyperparams.eps = testing_eps
        std_acc = hmc.test_hmc_with_average_logits(TEST_DATA, posterior_samples)
        ibp_acc = ibp_eval(conv_net, hyperparams, TEST_DATA, posterior_samples)
        id_auroc = auroc(conv_net, TEST_DATA, posterior_samples)
        ood_auroc = ood_detection_auc_and_ece(conv_net, TEST_DATA, ood_data_test, posterior_samples)[0]
        result_dict["num_epochs"].append({"k": k,
                                   "std_acc": std_acc,
                                   "ibp_acc": ibp_acc,
                                   "id_auroc": id_auroc,
                                   "ood_auroc": ood_auroc})
        print(f"Finished for k={k}")

    with open("experiments/privacy_study.yaml", "w", encoding="utf-8") as f:
        yaml.dump(result_dict, f)

def ablation_study():
    ood_data_test = load_fashion_mnist()[1]
    # Run the ablation study
    conv_net = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(
        num_epochs=60, num_burnin_epochs=25, step_size=0.01, lf_steps=120, batch_size=500, num_chains=3, decay_epoch_start=50,
        lr_decay_magnitude=0.5, warmup_step_size=0.2, momentum_std=0.001, prior_mu=0.0, prior_std=15, alpha_warmup_epochs=16,
        eps_warmup_epochs=20, alpha=0.993, alpha_pre_trained=0.75, step_size_pre_trained=0.001, eps=0.075,
        run_dp=True, grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=0.05, tau_l=0.05
    )

    testing_eps = 0.075
    epsilons = [0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    clip_bounds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01]
    result_dict = {"eps": [], "dp": []}
    for curr_eps in epsilons:
        # update hps
        hyperparams.eps = curr_eps
        hmc, posterior_samples = run_experiment_adv_hmc(conv_net, TRAIN_DATA, hyperparams, AttackType.IBP, init_from_trained=True)
        hyperparams.eps = testing_eps
        std_acc = hmc.test_hmc_with_average_logits(TEST_DATA, posterior_samples)
        ibp_acc = ibp_eval(conv_net, hyperparams, TEST_DATA, posterior_samples)
        id_auroc = auroc(conv_net, TEST_DATA, posterior_samples)
        ood_auroc = ood_detection_auc_and_ece(conv_net, TEST_DATA, ood_data_test, posterior_samples)[0]
        result_dict["eps"].append({"value": curr_eps,
                                   "std_acc": std_acc,
                                   "ibp_acc": ibp_acc,
                                   "id_auroc": id_auroc,
                                   "ood_auroc": ood_auroc})
        print(f"Finished for epsilon: {curr_eps}")
    hyperparams.eps = testing_eps
    for clip_bound in clip_bounds:
        # update hps
        hyperparams.grad_clip_bound = clip_bound
        hmc, posterior_samples = run_experiment_adv_hmc(conv_net, TRAIN_DATA, hyperparams, AttackType.IBP, init_from_trained=True)
        std_acc = hmc.test_hmc_with_average_logits(TEST_DATA, posterior_samples)
        ibp_acc = ibp_eval(conv_net, hyperparams, TEST_DATA, posterior_samples)
        id_auroc = auroc(conv_net, TEST_DATA, posterior_samples)
        ood_auroc = ood_detection_auc_and_ece(conv_net, TEST_DATA, ood_data_test, posterior_samples)[0]
        result_dict["dp"].append({"value": clip_bound,
                                  "std_acc": std_acc,
                                  "ibp_acc": ibp_acc,
                                  "id_auroc": id_auroc,
                                  "ood_auroc": ood_auroc})
        print(f"Finished for clip bound: {clip_bound}")
    with open("experiments/ablation_mnist.yaml", "w", encoding="utf-8") as f:
        yaml.dump(result_dict, f)

def membership_inference_bnn_experiment(adv: False, load_from_file: False):
    moments = get_marginal_distributions(TRAIN_DATA)
    target_network = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=20, num_burnin_epochs=5, step_size=0.01, lf_steps=120, criterion=torch.nn.CrossEntropyLoss(),
                                 batch_size=100, momentum_std=0.01)

    hmc, posterior_samples = AdvHamiltonianMonteCarlo(target_network, hyperparams, AttackType.IBP), None
    if load_from_file:
        savedir = "experiments/posterior_samples/ibp_hmc_mnist/"
        if adv:
            savedir = "experiments/posterior_samples/ibp_hmc_dp_mnist/"
        posterior_samples = load_samples(savedir)
    else:
        hmc, posterior_samples = run_experiment_hmc(target_network, TRAIN_DATA, hyperparams)
    accuracy = hmc.test_hmc_with_average_logits(TRAIN_DATA, posterior_samples)
    print(f"Accuracy of BNN with average logits on training set: {accuracy} %")

    run_bnn_membership_inference_attack(TRAIN_DATA, target_network, moments, posterior_samples)

TRAIN_DATA, TEST_DATA = load_mnist()
# adv_dp_experiment(write_results=True, save_model=True, for_adv_comparison=False)
# hmc_dp_experiment(write_results=False, for_adv_comparison=False, save_model=False)
# dnn_experiment(save_model=False, write_results=False, for_adv_comparison=False)
# ablation_study()
privacy_study()
