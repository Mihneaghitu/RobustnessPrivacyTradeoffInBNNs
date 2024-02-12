import sys

import torch

# can only run the script from this dir
sys.path.append('../probabilistic')
sys.path.append('../')

from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.HMC.bnn import VanillaBNN
from probabilistic.HMC.hmc import HamiltonianMonteCarlo, HyperparamsHMC
from probabilistic.HMC.losses import (cross_entropy_likelihood,
                                      neg_log_normal_pdf)


def run_experiment():
    hps = HyperparamsHMC(num_epochs=10,
                         num_burnin_epochs=3,
                         lf_step=0.002,
                         momentum_var=torch.tensor(1.0),
                         prior_mu=torch.tensor(0.0),
                         prior_var=torch.tensor(0.5),
                         steps_per_epoch=20,
                         batch_size=1000)

    likelihood = cross_entropy_likelihood
    prior = neg_log_normal_pdf
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    net = VanillaBNN().to(TORCH_DEVICE)

    hmc = HamiltonianMonteCarlo(hps, net, loss_fn, likelihood, prior)
    train_data, test_data = load_mnist()

    hmc_samples = hmc.train_hmc(train_data)

    accuracy = hmc.test_hmc(hmc_samples, test_data)
    print(f'Accuracy: {accuracy}')


WITH_DP = False
if sys.argv[1] == 'dp':
    WITH_DP = True

run_experiment()