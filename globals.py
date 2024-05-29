from enum import Enum

import torch


class LoggerType(Enum):
    STDOUT = 1
    WANDB = 2

cuda_available = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda' if cuda_available else 'cpu')
LOGGER_TYPE = LoggerType.STDOUT
ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42
torch.manual_seed(ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING)
torch.backends.cudnn.deterministic = True

# models for experiments
class AdvModel(Enum):
    SGD = 0
    HMC = 1
    FGSM_HMC = 2
    PGD_HMC = 3
    IBP_HMC = 4
class AdvDpModel(Enum):
    SGD = 0
    HMC = 1
    HMC_DP = 2
    IBP_HMC = 3
    IBP_HMC_DP = 4
ROOT_FNAMES_ADV = ["vanilla_network", "hmc_", "fgsm_hmc_", "pgd_hmc_", "ibp_hmc_"]
MODEL_NAMES_ADV = ["SGD", "HMC", "ADV-HMC (FGSM)", "ADV-HMC (PGD)", "ADV-HMC (IBP)"]
ROOT_FNAMES_ADV_DP = ["vanilla_network", "hmc_", "hmc_dp_", "ibp_hmc_", "ibp_hmc_dp_"]
MODEL_NAMES_ADV_DP = ["SGD", "HMC", "HMC-DP", "ADV-HMC (IBP)", "ADV-DP-HMC (IBP)"]
# metrics for plotting
ACCURACY_METRICS = ["STD_ACC", "FGSM_ACC", "PGD_ACC", "IBP_ACC"]
UNCERTAINTY_METRICS = ["IN_DISTRIB_AUROC", "IN_DISTRIB_ECE", "OOD_AUROC", "OOD_ECE"]
ACCURACY_METRICS_NAMES = [r"\textbf{STANDARD ACCURACY}",
                          r"\textbf{ACCURACY UNDER FGSM ATTACKS ($\mathbf{\epsilon=0.01}$)}",
                          r"\textbf{ACCURACY UNDER PGD ATTACKS ($\mathbf{\epsilon=0.01}$)}",
                          r"\textbf{CERTIFIED IBP ROBUSTNESS ($\mathbf{\epsilon=0.01}$)}"]
MODEL_PLOT_NAMES_ADV = [r"\textbf{SGD}", r"\textbf{HMC}", r"\textbf{FGSM-HMC}", r"\textbf{PGD-HMC}", r"\textbf{IBP-HMC}"]
MODEL_PLOT_NAMES_ADV_DP = [r"\textbf{SGD}", r"\textbf{HMC}", r"\textbf{DP-HMC}", r"\textbf{IBP-HMC}", r"\textbf{IBP-DP-HMC}"]
UNCERTAINTY_METRICS_NAMES = [r"\textbf{IN-DISTRIBUTION AUROC}",
                             r"\textbf{IN-DISTRIBUTION ECE}",
                             r"\textbf{OUT-OF-DISTRIBUTION AUROC}",
                             r"\textbf{OUT-OF-DISTRIBUTION ECE}"]
