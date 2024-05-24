import torch

from common.attack_types import AttackType
from common.dataset_utils import load_mnist
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import VanillaNetMnist
from experiments.experiment_utils import (compute_metrics_hmc,
                                          compute_metrics_sgd,
                                          run_experiment_adv_hmc,
                                          run_experiment_hmc,
                                          run_experiment_sgd)
from globals import TORCH_DEVICE
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import VanillaBnnMnist


def adv_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    # print the seed
    print(f"Using device: {TORCH_DEVICE}")
    vanilla_bnn = VanillaBnnMnist().to(TORCH_DEVICE)
    train_data, test_data = load_mnist()
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

    hmc, posterior_samples = run_experiment_adv_hmc(vanilla_bnn, train_data, hyperparams, attack_type, fname, init_from_trained=True)
    compute_metrics_hmc(hmc ,test_data, posterior_samples, testing_eps=0.05, write_results=write_results,
                        model_name=model_name, dset_name="MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)


def hmc_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    train_data, test_data = load_mnist()
    print("Training Bayesian Neural Network using HMC...")
    target_network = VanillaBnnMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=70, num_burnin_epochs=10, step_size=0.0165, lf_steps=120, num_chains=3,
                                 batch_size=500, momentum_std=0.01, decay_epoch_start=35, lr_decay_magnitude=0.5,
                                 run_dp=True, grad_clip_bound=0.4, acceptance_clip_bound=0.4, tau_g=0.2, tau_l=0.2)

    fname = f'''hmc{"_dp" if hyperparams.run_dp else ""}_mnist''' if save_model else None
    hmc, posterior_samples = run_experiment_hmc(target_network, train_data, hyperparams, save_dir_name=fname)
    model_name = "HMC-DP" if hyperparams.run_dp else "HMC"
    compute_metrics_hmc(hmc ,test_data, posterior_samples, testing_eps=0.075, write_results=write_results,
                        model_name=model_name, dset_name="MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)

def dnn_experiment(write_results: bool = False, save_model: bool = False, for_adv_comparison: bool = False, pre_train: bool = False):
    train, test = load_mnist()
    net = VanillaNetMnist().to(TORCH_DEVICE)
    hyperparams = Hyperparameters(num_epochs=0, lr=0.1, batch_size=60, lr_decay_magnitude=0.5, decay_epoch_start=20,
                                  alpha=0.5, eps=0.1, eps_warmup_itrs=6000, alpha_warmup_itrs=10000, warmup_itr_start=3000,
                                  run_dp=True, grad_norm_bound=0.5, dp_sigma=0.1)

    fname = "vanilla_network.pt" if save_model else None
    attack_type = AttackType.IBP
    pipeline = run_experiment_sgd(net, train, hyperparams, attack_type=attack_type, save_file_name=fname)
    if attack_type == AttackType.IBP and pre_train:
        name_for_bnn_init = "vanilla_network_ibp" if not hyperparams.run_dp else "vanilla_network_ibp_dp"
        torch.save(pipeline.net.state_dict(), f"pre_trained/{name_for_bnn_init}.pt")

    compute_metrics_sgd(pipeline, test, testing_eps=0.075, write_results=write_results, dset_name="MNIST", for_adv_comparison=for_adv_comparison)


# adv_dp_experiment(write_results=True, save_model=True, for_adv_comparison=False)
# hmc_dp_experiment(write_results=False, for_adv_comparison=False, save_model=False)
# dnn_experiment(save_model=False, write_results=False, for_adv_comparison=False)
