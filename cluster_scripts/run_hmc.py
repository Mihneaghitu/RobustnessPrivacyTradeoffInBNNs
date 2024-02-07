import sys
from typing import List

import torch

# can only run the script from this dir
sys.path.append('../probabilistic')
sys.path.append('../')

from dataset_utils import load_mnist
from globals import TORCH_DEVICE
from probabilistic.bnn import BNN, VanillaBNN
from probabilistic.hamiltonian import Hamiltonian, HyperparamsHMC
from probabilistic.models import WeightModel
from probabilistic.pipeline import hmc, test_hmc


def train_hmc(bnn_net: BNN, dp: bool = False) -> List[torch.Tensor]:
    # First create the BNN
    train_data, _ = load_mnist("..")
    # should broadcast to the correct shape
    w_prior_mean, w_prior_cov = torch.zeros(1), torch.eye(1)
    weight_probabilistic_model = WeightModel(prior_mean=w_prior_mean, prior_variance=w_prior_cov, ll_variance=torch.tensor(0.1))
    w_prior_func = weight_probabilistic_model.log_gaussian_prior
    w_likelihood_func = weight_probabilistic_model.log_gaussian_likelihood

    momentum_variances = torch.ones(1)
    hamiltonian = Hamiltonian(w_prior_func, w_likelihood_func, momentum_variances, train_data, net=bnn_net, batch_size=2500)

    hps = HyperparamsHMC(num_epochs=1500, num_burnin_epochs=150, lf_step=0.0001, steps_per_epoch=30)

    if dp:
        hps.gradient_norm_bound = 0.5
    param_samples = hmc(hamiltonian, hps, dp=dp)

    print(param_samples[:20] + param_samples[500:520] + param_samples[1000:1020] + param_samples[1470:1495])
    return param_samples


vanilla_bnn = VanillaBNN()
vanilla_bnn.to(TORCH_DEVICE)

WITH_DP = False
if sys.argv[1] == 'dp':
    WITH_DP = True

samples = train_hmc(vanilla_bnn, dp=WITH_DP)
print(f'Number of samples: {len(samples)}')
test_data = load_mnist()[1]
accuracy = test_hmc(vanilla_bnn, samples, test_data)
print(f'Accuracy: {accuracy}')
