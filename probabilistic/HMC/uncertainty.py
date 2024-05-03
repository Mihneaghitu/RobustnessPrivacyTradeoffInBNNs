import copy
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (MulticlassAUROC,
                                         MulticlassCalibrationError)

sys.path.append('../../')

from globals import TORCH_DEVICE
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


def auroc(net: VanillaBnnLinear, test_set: Dataset, posterior_samples: List[torch.Tensor]) -> float:
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    mean_posterior_predictive_distribution = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        batch_mean_ppd = net.hmc_forward(batch_data.to(TORCH_DEVICE), posterior_samples, lambda x: F.softmax(x, dim=1))
        mean_posterior_predictive_distribution = torch.cat((mean_posterior_predictive_distribution, batch_mean_ppd), dim=0)

    mc_auroc = MulticlassAUROC(num_classes=10, average='macro', thresholds=None)
    auc_val = mc_auroc(mean_posterior_predictive_distribution, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    return auc_val

def ece(net: VanillaBnnLinear, test_set: Dataset, posterior_samples: List[torch.Tensor]):
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    mean_posterior_predictive_distribution = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        batch_mean_ppd = net.hmc_forward(batch_data.to(TORCH_DEVICE), posterior_samples, lambda x: F.softmax(x, dim=1))
        mean_posterior_predictive_distribution = torch.cat((mean_posterior_predictive_distribution, batch_mean_ppd), dim=0)

    mc_ece = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')
    ece_val = mc_ece(mean_posterior_predictive_distribution, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    return ece_val
