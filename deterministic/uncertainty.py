import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (MulticlassAUROC,
                                         MulticlassCalibrationError)

sys.path.append('../')

from sklearn.metrics import roc_auc_score

from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE


def auroc(net: VanillaNetLinear, test_set: Dataset) -> float:
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    y_pred = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        y_pred_batch = net(batch_data.to(TORCH_DEVICE))
        y_pred_batch = F.softmax(y_pred_batch, dim=1)
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

    mc_auroc = MulticlassAUROC(num_classes=10, average='macro', thresholds=None)
    auc_val = mc_auroc(y_pred, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    # ---------------------- Sanity check ----------------------
    y_pred = y_pred.clone().detach().cpu().numpy()
    y_true = copy.deepcopy(test_set.targets).clone().detach().cpu().numpy()
    alt_auroc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

    assert np.isclose(auc_val.item(), alt_auroc, atol=1e-5)
    # ----------------------------------------------------------

    return auc_val

def ece(net: VanillaNetLinear, test_set: Dataset):
    net.eval()
    data_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    y_pred = torch.tensor([], dtype=torch.float32, device=TORCH_DEVICE)
    for batch_data, _ in data_loader:
        y_pred_batch = net(batch_data.to(TORCH_DEVICE))
        y_pred_batch = F.softmax(y_pred_batch, dim=1)
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

    mc_ece = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')
    ece_val = mc_ece(y_pred, copy.deepcopy(test_set.targets).to(TORCH_DEVICE))

    return ece_val
