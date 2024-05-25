
import torch
from torch.nn import BCEWithLogitsLoss

from common.attack_types import AttackType
from common.dataset_utils import load_pneumonia_mnist
from deterministic.hyperparams import Hyperparameters
from deterministic.vanilla_net import ConvNetPneumoniaMnist
from experiments.experiment_utils import (compute_metrics_hmc,
                                          compute_metrics_sgd,
                                          run_experiment_adv_hmc,
                                          run_experiment_hmc,
                                          run_experiment_sgd)
from globals import TORCH_DEVICE
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import ConvBnnPneumoniaMnist


def adv_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    # print the seed
    print(f"Using device: {TORCH_DEVICE}")
    vanilla_bnn = ConvBnnPneumoniaMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=80, num_burnin_epochs=25, step_size=0.04, warmup_step_size=0.25, lf_steps=24, batch_size=218,
                    num_chains=3, momentum_std=0.002, alpha=0.975, alpha_pre_trained=0.75, eps=0.01, step_size_pre_trained=0.001,
                    decay_epoch_start=40, lr_decay_magnitude=0.5, eps_warmup_epochs=20, alpha_warmup_epochs=16, run_dp=True,
                    grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1, prior_std=5, criterion=BCEWithLogitsLoss())

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

    hmc, posterior_samples = run_experiment_adv_hmc(vanilla_bnn, TRAIN_DATA, hyperparams, attack_type, fname, init_from_trained=False)
    compute_metrics_hmc(hmc, TEST_DATA, posterior_samples, testing_eps=0.01, write_results=write_results,
                        model_name=model_name, dset_name="PNEUMONIA_MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)

def hmc_dp_experiment(write_results: bool = False, for_adv_comparison: bool = True, save_model: bool = False):
    target_network = ConvBnnPneumoniaMnist().to(TORCH_DEVICE)
    hyperparams = HyperparamsHMC(num_epochs=60, num_burnin_epochs=8, step_size=0.07, lf_steps=24, criterion=BCEWithLogitsLoss(),
                                 num_chains=3, batch_size=218, momentum_std=0.005, decay_epoch_start=40, lr_decay_magnitude=0.5,
                                 run_dp=True, grad_clip_bound=0.5, acceptance_clip_bound=0.5, tau_g=0.1, tau_l=0.1)

    fname = f'''hmc{"_dp" if hyperparams.run_dp else ""}_pneumonia_mnist''' if save_model else None
    hmc, posterior_samples = run_experiment_hmc(target_network, TEST_DATA, hyperparams, save_dir_name=fname)
    model_name = "HMC-DP" if hyperparams.run_dp else "HMC"
    compute_metrics_hmc(hmc, TEST_DATA, posterior_samples, testing_eps=0.01, write_results=write_results,
                        model_name=model_name, dset_name="PNEUMONIA_MNIST", for_adv_comparison=for_adv_comparison and not hyperparams.run_dp)

def dnn_experiment(write_results: bool = False, save_model: bool = False, for_adv_comparison: bool = False, pre_train: bool = False):
    net = ConvNetPneumoniaMnist().to(TORCH_DEVICE)
    hyperparams = Hyperparameters(num_epochs=98, lr=0.09, batch_size=109, lr_decay_magnitude=0.005, decay_epoch_start=25,
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

TRAIN_DATA, TEST_DATA = load_pneumonia_mnist()
# adv_dp_experiment(write_results=True, save_model=False, for_adv_comparison=True)
hmc_dp_experiment(write_results=True, for_adv_comparison=False, save_model=False)
# dnn_experiment(save_model=True, write_results=True, for_adv_comparison=True)
