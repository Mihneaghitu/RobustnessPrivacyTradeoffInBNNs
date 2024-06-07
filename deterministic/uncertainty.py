import copy
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (BinaryAUROC, BinaryCalibrationError,
                                         MulticlassAUROC,
                                         MulticlassCalibrationError)

sys.path.append('../')

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE


def auroc(net: VanillaNetLinear, test_set: Dataset) -> float:
    net.eval()
    to_prob = lambda x: F.softmax(x, dim=1) if net.get_num_classes() > 2 else torch.sigmoid(x)
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    y_pred = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        y_pred_batch = net(batch_data.to(TORCH_DEVICE))
        y_pred_batch = to_prob(y_pred_batch)
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

    mc_auroc = None
    if net.get_num_classes() > 1:
        mc_auroc = MulticlassAUROC(num_classes=10, average='macro', thresholds=None)
    else: # Binary classification
        mc_auroc = BinaryAUROC(thresholds=None)

    auc_val = mc_auroc(y_pred, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    # ---------------------- Sanity check ----------------------
    y_pred = y_pred.clone().detach().cpu().numpy()
    y_true = copy.deepcopy(test_set.targets).clone().detach().cpu().numpy()
    alt_auroc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

    assert np.isclose(auc_val.item(), alt_auroc, atol=1e-5)
    # ----------------------------------------------------------

    return round(float(auc_val.item()), 4)

def ece(net: VanillaNetLinear, test_set: Dataset) -> float:
    net.eval()
    to_prob = lambda x: F.softmax(x, dim=1) if net.get_num_classes() > 2 else torch.sigmoid(x)
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    y_pred = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        y_pred_batch = net(batch_data.to(TORCH_DEVICE))
        y_pred_batch = to_prob(y_pred_batch)
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

    mc_ece = None
    if net.get_num_classes() > 1:
        mc_ece = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')
    else:
        mc_ece = BinaryCalibrationError(n_bins=10, norm='l1')

    ece_val = mc_ece(y_pred, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    return round(float(ece_val.item()), 4)

def ood_detection_auc_and_ece(net: VanillaNetLinear, id_data: Dataset, ood_data: Dataset) -> Tuple[float, float]:
    to_prob = lambda x: F.softmax(x, dim=1) if net.get_num_classes() > 2 else torch.sigmoid(x)
    net.eval()
    id_data_loader = DataLoader(id_data, batch_size=1000, shuffle=False)
    ood_data_loader = DataLoader(ood_data, batch_size=1000, shuffle=False)
    id_pred = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    ood_pred = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for (id_batch_data, _), (ood_batch_data, _) in zip(id_data_loader, ood_data_loader):
        id_pred_batch = to_prob(net(id_batch_data.to(TORCH_DEVICE)))
        ood_pred_batch = to_prob(net(ood_batch_data.to(TORCH_DEVICE)))
        id_pred = torch.cat((id_pred, id_pred_batch), dim=0)
        ood_pred = torch.cat((ood_pred, ood_pred_batch), dim=0)

    threshold = 0.5 if net.get_num_classes() > 2 else 0.65
    id_pred_probs, ood_pred_probs = torch.max(id_pred, dim=1).values, torch.max(ood_pred, dim=1).values
    if net.get_num_classes() <= 2:
        id_pred_probs = torch.maximum(id_pred, 1 - id_pred)
        ood_pred_probs = torch.maximum(ood_pred, 1 - ood_pred)
    y_pred = torch.cat((id_pred_probs, ood_pred_probs), dim=0)
    y_pred = torch.where(y_pred > threshold, torch.ones_like(y_pred), torch.zeros_like(y_pred)) # if less than threshold, ood, else id
    y_true = torch.cat((torch.ones_like(id_pred_probs), torch.zeros_like(ood_pred_probs)), dim=0).to(TORCH_DEVICE)
    y_pred, y_true = y_pred.clone().detach().cpu().numpy(), y_true.clone().detach().cpu().numpy()

    # ---- AUROC ----
    auc_val = roc_auc_score(y_true, y_pred, average='macro')
    # ---- ECE ----
    ece_val = calibration_curve(y_true, y_pred, n_bins=2, strategy='quantile')[1].mean()

    return round(float(auc_val), 4), round(float(ece_val), 4)
