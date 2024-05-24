import copy
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (BinaryAUROC, BinaryCalibrationError,
                                         MulticlassAUROC,
                                         MulticlassCalibrationError)

sys.path.append('../../')

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from globals import TORCH_DEVICE
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


def auroc(net: VanillaBnnLinear, test_set: Dataset, posterior_samples: List[torch.Tensor]) -> float:
    net.eval()
    last_layer_act = lambda x: F.softmax(x, dim=1) if net.get_num_classes() > 2 else torch.sigmoid(x)
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    mean_posterior_predictive_distribution = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        batch_mean_ppd = net.hmc_forward(batch_data.to(TORCH_DEVICE), posterior_samples, last_layer_act)
        mean_posterior_predictive_distribution = torch.cat((mean_posterior_predictive_distribution, batch_mean_ppd), dim=0)

    mc_auroc = None
    if net.get_num_classes() > 2:
        mc_auroc = MulticlassAUROC(num_classes=10, average='macro', thresholds=None)
    else:
        mc_auroc = BinaryAUROC(thresholds=None)
    auc_val = mc_auroc(mean_posterior_predictive_distribution, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    # ---------------------- Sanity check ----------------------
    y_pred = mean_posterior_predictive_distribution.clone().detach().cpu().numpy()
    y_true = copy.deepcopy(test_set.targets).clone().detach().cpu().numpy()
    alt_auroc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

    assert np.isclose(auc_val.item(), alt_auroc, atol=1e-5)
    # ----------------------------------------------------------

    return round(float(auc_val), 4)

def ece(net: VanillaBnnLinear, test_set: Dataset, posterior_samples: List[torch.Tensor]) -> float:
    net.eval()
    last_layer_act = lambda x: F.softmax(x, dim=1) if net.get_num_classes() > 2 else torch.sigmoid(x)
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    mean_posterior_predictive_distribution = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        batch_mean_ppd = net.hmc_forward(batch_data.to(TORCH_DEVICE), posterior_samples, last_layer_act)
        mean_posterior_predictive_distribution = torch.cat((mean_posterior_predictive_distribution, batch_mean_ppd), dim=0)

    mc_ece = None
    if net.get_num_classes() > 2:
        mc_ece = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')
    else:
        mc_ece = BinaryCalibrationError(n_bins=10, norm='l1')
    ece_val = mc_ece(mean_posterior_predictive_distribution, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    return round(float(ece_val), 4)

def ood_detection_auc_and_ece(net: VanillaBnnLinear, id_data: Dataset, ood_data: Dataset, posterior_samples: List[torch.Tensor]) -> Tuple[float, float]:
    net.eval()
    last_layer_act = lambda x: F.softmax(x, dim=1) if net.get_num_classes() > 2 else torch.sigmoid(x)
    id_data_loader = DataLoader(id_data, batch_size=1000, shuffle=False)
    ood_data_loader = DataLoader(ood_data, batch_size=1000, shuffle=False)
    id_mean_ppd = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    ood_mean_ppd = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for (id_batch_data, _), (ood_batch_data, _) in zip(id_data_loader, ood_data_loader):
        id_batch_mean_ppd = net.hmc_forward(id_batch_data.to(TORCH_DEVICE), posterior_samples, last_layer_act)
        ood_batch_mean_ppd = net.hmc_forward(ood_batch_data.to(TORCH_DEVICE), posterior_samples, last_layer_act)
        id_mean_ppd = torch.cat((id_mean_ppd, id_batch_mean_ppd), dim=0)
        ood_mean_ppd = torch.cat((ood_mean_ppd, ood_batch_mean_ppd), dim=0)

    id_pred_probs, ood_pred_probs = torch.max(id_mean_ppd, dim=1).values, torch.max(ood_mean_ppd, dim=1).values
    y_pred = torch.cat((id_pred_probs, ood_pred_probs), dim=0)
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32), torch.zeros_like(y_pred, dtype=torch.float32))
    #! I.e. the softmax probabilities of the predicted classes should be close to 1 for in-distribution data and close to 0 for OOD data
    y_true = torch.cat((torch.ones_like(id_pred_probs), torch.zeros_like(ood_pred_probs)), dim=0).to(TORCH_DEVICE, dtype=torch.int64)
    y_pred, y_true = y_pred.clone().detach().cpu().numpy(), y_true.clone().detach().cpu().numpy()

    # ---- AUROC ----
    auc_val = roc_auc_score(y_true, y_pred, average='macro')
    # ---- ECE ----
    ece_val = calibration_curve(y_true, y_pred, n_bins=2, strategy='quantile')[1].mean()

    return round(float(auc_val), 4), round(float(ece_val), 4)


def ibp_auc_and_ece(net: VanillaBnnLinear, test_set: Dataset, posterior_samples: List[torch.Tensor], eps: float) -> Tuple[float]:
    net.eval()
    data_loader = DataLoader(test_set, batch_size=100, shuffle=False)
    mean_ppd = torch.zeros((len(test_set), net.get_output_size()), dtype=torch.float32, device=TORCH_DEVICE)
    for posterior_sample in posterior_samples:
        net.set_params(posterior_sample)
        sample_mean_ppd = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
        for batch_data, batch_targets in data_loader:
            batch_mean_ppd = net.get_worst_case_logits(batch_data.to(TORCH_DEVICE), batch_targets.to(TORCH_DEVICE), eps)
            sample_mean_ppd = torch.cat((sample_mean_ppd, batch_mean_ppd), dim=0)
        mean_ppd += (sample_mean_ppd / len(posterior_samples))

    mean_ppd = F.softmax(mean_ppd, dim=1)

    # ------------------------- AUROC --------------------------
    mc_auroc = MulticlassAUROC(num_classes=10, average='macro', thresholds=None)
    ibp_auc = mc_auroc(mean_ppd, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))
    # Sanity check
    y_pred = mean_ppd.clone().detach().cpu().numpy()
    y_true = copy.deepcopy(test_set.targets).clone().detach().cpu().numpy()
    alt_auroc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

    assert np.isclose(ibp_auc.item(), alt_auroc, atol=1e-5)
    # ----------------------------------------------------------
    # ------------------------- ECE ----------------------------
    mc_ece = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')
    ibp_ece = mc_ece(mean_ppd, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))
    # ----------------------------------------------------------

    return ibp_auc, ibp_ece
