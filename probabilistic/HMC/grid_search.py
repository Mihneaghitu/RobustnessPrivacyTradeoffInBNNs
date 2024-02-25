import copy
import os
import sys

import torch
import wandb

sys.path.append('../../')
from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.vanilla_bnn import HyperparamsHMC, VanillaBnnLinear

print(f"Using device: {TORCH_DEVICE}")
VANILLA_BNN = VanillaBnnLinear().to(TORCH_DEVICE)
max_predictive_acc, optimal_samples, best_cnt = 0, None, 0
train_data, test_data = load_mnist("../")

def default_config() -> dict:
    no_dp_config = {
        'method': 'bayes',
        'metric': {
            'name': 'acc_with_average_logits',
            'goal': 'maximize'
        }
    }
    params_no_dp_dict = {
        "num_epochs": {
            'values': [250]
        },
        "num_burnin_epochs": {
            'values': [50]
        },
        "batch_size": {
            'values':  [64, 128, 256, 512]
        },
        "step_size": {
            'values': [0.01, 0.05, 0.1]
        },
        "lf_steps": {
            'values': [25, 50, 75, 100]
        },
        "momentum_std": {
            'values': [0.01, 0.05, 0.1]
        },
    }
    no_dp_config['parameters'] = params_no_dp_dict

    return no_dp_config

def dp_config() -> dict:
    with_dp_config = {
        'method': 'bayes',
        'metric': {
            'name': 'acc_with_average_logits',
            'goal': 'maximize'
        }
    }
    params_dp_dict = {
        "num_epochs": {
            'values': [250]
        },
        "num_burnin_epochs": {
            'values': [50]
        },
        "batch_size": {
            'values':  [64, 128, 256, 512]
        },
        "step_size": {
            'values': [0.01, 0.05, 0.1]
        },
        "lf_steps": {
            'values': [25, 50, 75, 100]
        },
        "momentum_std": {
            'values': [0.01, 0.05, 0.1]
        },
        "grad_norm_bound": {
            'values': [0.1, 0.5, 1.0, 5.0]
        },
        "dp_sigma": {
            'values': [0.005, 0.01, 0.05, 0.1]
        }
    }
    with_dp_config['parameters'] = params_dp_dict

    return with_dp_config


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- Grid Search Without Differential Privacy --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def grid_search_no_dp():
    # NOTE: lf_steps = len(train_data) // batch_size = 60000 // batch_size = 468 -- this is to see all the dataset for one numerical integration step
    with wandb.init():
        grid_config = wandb.config
        hyperparams = HyperparamsHMC(
            num_epochs=grid_config.num_epochs,
            num_burnin_epochs=grid_config.num_burnin_epochs,
            step_size=grid_config.step_size,
            lf_steps=grid_config.lf_steps,
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=grid_config.batch_size,
            momentum_std=grid_config.momentum_std
        )
        hmc = HamiltonianMonteCarlo(VANILLA_BNN, hyperparams)

        # Train and test
        samples = hmc.train_mnist_vanilla(train_data)
        acc_with_average_logits = hmc.test_hmc_with_average_logits(test_data, samples)
        global max_predictive_acc, optimal_samples, best_cnt
        if acc_with_average_logits > max_predictive_acc:
            max_predictive_acc = acc_with_average_logits
            optimal_samples = copy.deepcopy(samples)
        if acc_with_average_logits > 92.0:
            torch.save(optimal_samples, os.path.join(wandb.run.dir, f"optimal_samples_over_92_{best_cnt}"))
            best_cnt += 1

        print(f'Accuracy with average logits: {acc_with_average_logits} %')



# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- Grid Search With Differential Privacy -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def grid_search_with_dp():
    with wandb.init():
        grid_config = wandb.config
        hyperparams = HyperparamsHMC(
            num_epochs=grid_config.num_epochs,
            num_burnin_epochs=grid_config.num_burnin_epochs,
            step_size=grid_config.step_size,
            lf_steps=grid_config.lf_steps,
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=grid_config.batch_size,
            momentum_std=grid_config.momentum_std,
            run_dp=True,
            grad_norm_bound=grid_config.grad_norm_bound,
            dp_sigma=grid_config.dp_sigma
        )
        hmc = HamiltonianMonteCarlo(VANILLA_BNN, hyperparams)

        # Train and test
        samples = hmc.train_mnist_vanilla(train_data)
        acc_with_average_logits = hmc.test_hmc_with_average_logits(test_data, samples)
        global max_predictive_acc, optimal_samples, best_cnt
        if acc_with_average_logits > max_predictive_acc:
            max_predictive_acc = acc_with_average_logits
            optimal_samples = copy.deepcopy(samples)
        if acc_with_average_logits > 92.0:
            torch.save(optimal_samples, os.path.join(wandb.run.dir, f"optimal_samples_over_92_{best_cnt}"))
            best_cnt += 1

        print(f'Accuracy with average logits: {acc_with_average_logits} %')

def setup():
    wandb.login(key="6af656612e6115c4b189c6074dadbfc436f21439")

def run_no_dp_sweep():
    setup()
    sweep_config = default_config()

    global max_predictive_acc, optimal_samples
    max_predictive_acc, optimal_samples = 0, None
    sweep_id_1 = wandb.sweep(sweep_config, project="hmc_mnist_no_dp_bayes")
    wandb.agent(sweep_id=sweep_id_1, function=grid_search_no_dp)
    torch.save(optimal_samples, "optimal_samples_without_dp.pt")

def run_dp_sweep():
    setup()
    sweep_config = dp_config()

    global max_predictive_acc, optimal_samples
    max_predictive_acc, optimal_samples = 0, None

    sweep_id_2 = wandb.sweep(sweep_config, project="hmc_mnist_with_dp_bayes")
    wandb.agent(sweep_id=sweep_id_2, function=grid_search_with_dp)
    torch.save(optimal_samples, "optimal_samples_with_dp.pt")
