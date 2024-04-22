import sys

import torch
import wandb

sys.path.append('../../')
from functools import partial

import globals as glb
from dataset_utils import load_mnist
from globals import TORCH_DEVICE, LoggerType
from probabilistic.attack_types import AttackType
from probabilistic.HMC.adv_robust_dp_hmc import AdvHamiltonianMonteCarlo
from probabilistic.HMC.attacks import (fgsm_predictive_distrib_attack,
                                       ibp_eval, pgd_predictive_distrib_attack)
from probabilistic.HMC.hyperparams import HyperparamsHMC
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear

print(f"Using device: {TORCH_DEVICE}")
VANILLA_BNN = VanillaBnnLinear().to(TORCH_DEVICE)
glb.LOGGER_TYPE = LoggerType.WANDB
train_data, test_data = load_mnist("../../")

def adv_no_dp_config() -> dict:
    adv_no_dp_config = {
        'method': 'bayes',
        'metric': {
            'name': 'composite_std_robust_metric',
            'goal': 'maximize'
        }
    }
    params_adv_dict = {
        "num_epochs": {
            'values': [250]
        },
        "num_burnin_epochs": {
            'values': [50]
        },
        "batch_size": {
            'values':  [128, 256, 512]
        },
        "step_size": {
            'values': [0.01, 0.05, 0.1]
        },
        "lf_steps": {
            'values': [50, 100, 250, 500, 600]
        },
        "momentum_std": {
            'values': [0.01, 0.05, 0.1]
        },
        "alpha": {
            'values': [0.5, 0.6, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.965, 0.97, 0.975, 0.98]
        },
        "eps": {
            'values': [0.1]
        },
        "test_eps": {
            'values': [0.05]
        }
    }
    adv_no_dp_config['parameters'] = params_adv_dict

    return adv_no_dp_config

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- Grid Search Without Differential Privacy --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def grid_search_adv_no_dp(attack_type: str):
    with wandb.init(resume=True):
        print(wandb.config)
        grid_config = wandb.config
        hyperparams = HyperparamsHMC(
            num_epochs=grid_config.num_epochs,
            num_burnin_epochs=grid_config.num_burnin_epochs,
            step_size=grid_config.step_size,
            lf_steps=grid_config.lf_steps,
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=grid_config.batch_size,
            momentum_std=grid_config.momentum_std,
            run_dp=False,
            alpha=grid_config.alpha,
            eps=grid_config.eps
        )

        attack = None
        match attack_type:
            case "fgsm":
                attack = AttackType.FGSM
            case "ibp":
                attack = AttackType.IBP
            case _:
                attack = AttackType.PGD
        print(f"Attack type is {attack}")
        hmc = AdvHamiltonianMonteCarlo(VANILLA_BNN, hyperparams, attack)
        posterior_samples = hmc.train_mnist_vanilla(train_data)

        #^ Test epsilon needs to be smaller than training epsilon
        hmc.hps.eps = grid_config.test_eps
        adv_fgsm_test_set = fgsm_predictive_distrib_attack(hmc.net, hmc.hps, test_data, posterior_samples)
        adv_pgd_test_set = pgd_predictive_distrib_attack(hmc.net, hmc.hps, test_data, posterior_samples)

        acc_std = hmc.test_hmc_with_average_logits(test_data, posterior_samples)
        acc_fgsm = hmc.test_hmc_with_average_logits(adv_fgsm_test_set, posterior_samples)
        acc_pgd = hmc.test_hmc_with_average_logits(adv_pgd_test_set, posterior_samples)
        acc_ibp = ibp_eval(hmc.net, hmc.hps, test_data, posterior_samples)

        wandb.log({'acc_std': acc_std})
        wandb.log({'acc_fgsm': acc_fgsm})
        wandb.log({'acc_ibp': acc_ibp})
        wandb.log({'acc_pgd': acc_pgd})

        composite_std_robust_metric = 0
        if attack == AttackType.FGSM:
            composite_std_robust_metric += (acc_std / 3 + acc_fgsm)
        if attack == AttackType.IBP:
            composite_std_robust_metric += acc_ibp

        wandb.log({'composite_std_robust_metric': composite_std_robust_metric})

        print(f'Accuracy with average logits: {acc_std} %')
        print(f'Accuracy on FGSM adversarial test set: {acc_fgsm} %')
        print(f'Accuracy on PGD adversarial test set: {acc_pgd} %')
        print(f'Accuracy with IBP logits: {acc_ibp} %')

def setup():
    wandb.login(key="6af656612e6115c4b189c6074dadbfc436f21439")

def run_adv_no_dp_sweep(attack_type: str):
    setup()
    sweep_config = adv_no_dp_config()

    adv_no_dp_sweep = wandb.sweep(sweep=sweep_config, project="adv_robust_hmc_ibp")
    wandb.agent(sweep_id=adv_no_dp_sweep, function=partial(grid_search_adv_no_dp, attack_type), count=150)
