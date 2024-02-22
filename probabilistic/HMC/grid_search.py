import sys

import torch

import wandb

sys.path.append('../../')
from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.HMC.hmc import HamiltonianMonteCarlo
from probabilistic.HMC.vanilla_bnn import HyperparamsHMC, VanillaBnnLinear

wandb.login(key="6af656612e6115c4b189c6074dadbfc436f21439")
params_no_dp_dict = {
    "num_epochs": {
        'values': [150]
    },
    "num_burnin_epochs": {
        'values': [25]
    },
    "batch_size": {
        'values':  [32, 64, 128, 256, 512]
    },
    "step_size": {
        'values': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    },
    "lf_steps": {
        'values': [10, 25, 40, 50, 75, 100]
    },
    "momentum_std": {
        'values': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    },
}

sweep_config = {
    'method': 'grid',
}
sweep_config['parameters'] = params_no_dp_dict

print(f"Using device: {TORCH_DEVICE}")
VANILLA_BNN = VanillaBnnLinear().to(TORCH_DEVICE)

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
        train_data, test_data = load_mnist("../../")
        samples = hmc.train_mnist_vanilla(train_data)
        acc_with_average_logits = hmc.test_hmc_with_average_logits(test_data, samples)
        if acc_with_average_logits > max_mean_accuracy:
            max_mean_accuracy = acc_with_average_logits
        print(f'Accuracy with average logits: {acc_with_average_logits} %')


predictive_acc, optimal_samples = 0, None
sweep_id_1 = wandb.sweep(sweep_config, project="hmc_mnist_no_dp")
wandb.agent(sweep_id=sweep_id_1, function=grid_search_no_dp)
torch.save(optimal_samples, "optimal_samples_without_dp.pt")

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- Grid Search With Differential Privacy -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

params_with_dp_dict = params_no_dp_dict.copy()
params_with_dp_dict.update({
    "grad_norm_bound": {
        'values': [0.1, 0.5, 1.0, 2.0, 5.0]
    },
    "dp_sigma": {
        'values': [0.0005, 0.001, 0.005, 0.01, 0.025]
    }
})
sweep_config['parameters'] = params_with_dp_dict

def grid_search_with_dp():
    max_mean_accuracy, optimal_samples = 0, None
    with wandb.init():
        grid_config = wandb.config['parameters']
        print(grid_config)
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
        train_data, test_data = load_mnist("../../")
        samples = hmc.train_mnist_vanilla(train_data)
        acc_with_average_logits = hmc.test_hmc_with_average_logits(test_data, samples)
        if acc_with_average_logits > max_mean_accuracy:
            max_mean_accuracy = acc_with_average_logits
        print(f'Accuracy with average logits: {acc_with_average_logits} %')


predictive_acc, optimal_samples = 0, None
sweep_id_2 = wandb.sweep(sweep_config, project="hmc_mnist_with_dp")
wandb.agent(sweep_id=sweep_id_2, function=grid_search_with_dp)
torch.save(optimal_samples, "optimal_samples_with_dp.pt")
