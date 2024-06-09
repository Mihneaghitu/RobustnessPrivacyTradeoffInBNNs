import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from experiment_utils import get_delta_dp_bound

from globals import (ACCURACY_METRICS, ACCURACY_METRICS_NAMES,
                     MODEL_PLOT_NAMES_ADV, MODEL_PLOT_NAMES_ADV_DP,
                     UNCERTAINTY_METRICS, UNCERTAINTY_METRICS_NAMES)


#! 5 different models: dnn_sgd, hmc, fgsm_hmc, pgd_hmc, ibp_hmc
#! 8 metrics: std_acc, fgsm_acc, pgd_acc, ibp_acc, in_distrib_auroc, in_distrib_ece, ood_auroc, ood_ece
#? maybe privacy?
def plot_metrics(dataset_name: str = "MNIST", metric_type = "accuracy", for_adv_comparison: bool = True) -> None:
    plt.rcParams['text.usetex'] = True
    # now fill them up with the results from the yaml file
    fname = "/results_adv.yaml" if for_adv_comparison else "/results_all.yaml"
    adv_results_file = __file__.rsplit("/", 1)[0] + fname
    experiments_results = None
    with open(adv_results_file, 'r', encoding="utf-8") as file:
        experiments_results = yaml.safe_load(file)[dataset_name]

    models = list(experiments_results.keys())
    positions = np.arange(0, len(models), 1)
    colors = ["#d46d3d", "#e3c268", "#ccf56e", "#64d199", "#68a0e3"]
    hlines = np.arange(0, 1.1, 0.1)
    metrics = ACCURACY_METRICS if metric_type == "accuracy" else UNCERTAINTY_METRICS

    fig, ax = plt.subplots(2, 2, figsize=(30, 20), constrained_layout=True)
    for idx, metric in enumerate(metrics):
        row, col = idx // 2, idx % 2
        heights = [experiments_results[model][metric] for model in models]
        tick_labels = MODEL_PLOT_NAMES_ADV if for_adv_comparison else MODEL_PLOT_NAMES_ADV_DP
        ax[row, col].bar(positions, height=heights, width=0.65, align='center', color=colors, tick_label=tick_labels, alpha=0.9)
        ax[row, col].tick_params(axis='x', labelsize=14)
        ax[row, col].set_ylim(0, 1)
        ax[row, col].set_yticks(hlines)
        ax[row, col].hlines(hlines, -0.5, len(models) - 1 + 0.5, colors="gray", alpha=0.3)
        ax[row, col].set_title(ACCURACY_METRICS_NAMES[idx] if metric_type == "accuracy" else UNCERTAINTY_METRICS_NAMES[idx], fontsize=20)
        ax[row, col].set_ylabel("Accuracy", fontsize=13)
        ax[row, col].set_xlabel("Training method", fontsize=13)

    fig.show()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()

def plot_metrics_individual(dataset_name: str = "MNIST", metric_type = "accuracy", for_adv_comparison: bool = True) -> None:
    plt.rcParams['text.usetex'] = True
    # now fill them up with the results from the yaml file
    fname = "/results_adv.yaml" if for_adv_comparison else "/results_all.yaml"
    adv_results_file = __file__.rsplit("/", 1)[0] + fname
    experiments_results = None
    with open(adv_results_file, 'r', encoding="utf-8") as file:
        experiments_results = yaml.safe_load(file)[dataset_name]

    models = list(experiments_results.keys())
    positions = np.arange(0, len(models), 1)
    colors = ["#d46d3d", "#e3c268", "#ccf56e", "#64d199", "#68a0e3"]
    hlines = np.arange(0, 1.1, 0.1)
    metrics = ACCURACY_METRICS if metric_type == "accuracy" else UNCERTAINTY_METRICS
    metric_names = ACCURACY_METRICS_NAMES if metric_type == "accuracy" else UNCERTAINTY_METRICS_NAMES
    y_label = "Accuracy" if metric_type == "accuracy" else "Value"

    for idx, metric in enumerate(metrics):
        heights = [experiments_results[model][metric] for model in models]
        tick_labels = MODEL_PLOT_NAMES_ADV if for_adv_comparison else MODEL_PLOT_NAMES_ADV_DP
        plt.bar(positions, height=heights, width=0.65, align='center', color=colors, tick_label=tick_labels, alpha=0.9)
        plt.tick_params(axis='x', labelsize=31)
        plt.tick_params(axis='y', labelsize=14)
        plt.ylim(0, 1)
        plt.yticks(hlines)
        plt.hlines(hlines, -0.5, len(models) - 1 + 0.5, colors="gray", alpha=0.3)
        plt.title(metric_names[idx], fontsize=45)
        plt.ylabel(y_label, fontsize=30)
        plt.xlabel("Training method", fontsize=30)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()

def plot_ablation(dset_name: str = "MNIST") -> None:
    plt.rcParams['text.usetex'] = True
    fname = "ablation_" + dset_name.lower() + ".yaml"
    lims_dp, lims_eps = None, None
    if dset_name == "MNIST":
        lims_dp = {"acc": ((70, 90), (30, 60)), "unc": ((0.85, 1.01), (0.7, 1.0))}
        lims_eps = {"acc": ((75, 90), (30, 60)), "unc": ((0.85, 1.05), (0.7, 1.0))}
    else:
        lims_dp = {"acc": ((55, 90), (55, 75)), "unc": ((0.6, 1.05), (0.4, 0.85))}
        lims_eps = {"acc": ((50, 90), (50, 90)), "unc": ((0.4, 1.0), (0.2, 0.8))}
    with open(fname, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        dp_dicts = config["dp"]
        eps_dicts = config["eps"]

    dp_x_axis = np.flip([dp_dict["value"] for dp_dict in dp_dicts])
    dp_std_accs, dp_ibp_accs = np.flip([dp_dict["std_acc"] for dp_dict in dp_dicts]), np.flip([dp_dict["ibp_acc"] for dp_dict in dp_dicts])
    dp_id_auroc, dp_ood_auroc = np.flip([dp_dict["id_auroc"] for dp_dict in dp_dicts]), np.flip([dp_dict["ood_auroc"] for dp_dict in dp_dicts])

    titles_acc_dp = [r"\textbf{Standard Accuracy vs. Clip Bound Decrease}", r"\textbf{IBP Accuracy vs. Clip Bound Decrease}"]
    titles_acc_eps = [r"\textbf{Standard Accuracy vs. Epsilon Increase}", r"\textbf{IBP Accuracy vs. Epsilon Increase}"]
    titles_unc_dp = [r"\textbf{ID AUROC vs. Clip Bound Decrease}", r"\textbf{OOD AUROC vs. Clip Bound Decrease}"]
    titles_unc_eps = [r"\textbf{ID AUROC vs. Epsilon Increase}", r"\textbf{OOD AUROC vs. Epsilon Increase}"]
    __plot_ablation_accs(dp_std_accs, dp_ibp_accs, dp_x_axis, lims_dp["acc"], "Gradient Clip Bound", titles_acc_dp, reverse_x_axis=True)
    __plot_ablation_uncertainty(dp_id_auroc, dp_ood_auroc, dp_x_axis, "Gradient Clip Bound", lims_dp["unc"], titles_unc_dp, reverse_x_axis=True)

    eps_x_axis = [ed["value"] for ed in eps_dicts]
    eps_std_accs, eps_ibp_accs = [ed["std_acc"] for ed in eps_dicts], [ed["ibp_acc"] for ed in eps_dicts]
    eps_id_auroc, eps_ood_auroc = [ed["id_auroc"] for ed in eps_dicts], [ed["ood_auroc"] for ed in eps_dicts]

    __plot_ablation_accs(eps_std_accs, eps_ibp_accs, eps_x_axis, lims_eps["acc"], "Epsilon", titles_acc_eps)
    __plot_ablation_uncertainty(eps_id_auroc, eps_ood_auroc, eps_x_axis, "Epsilon", lims_eps["unc"], titles_unc_eps)

def __plot_ablation_accs(std_accs: np.ndarray, ibp_accs: np.ndarray, val_x_axis: np.ndarray, y_lims: tuple, x_label: str,
                         titles: list, reverse_x_axis: bool = False):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(val_x_axis, std_accs, label="Standard Accuracy", color="blue")
    axs[1].plot(val_x_axis, ibp_accs, label="IBP Accuracy", color="red")
    for ax, ttl, ylim in zip(axs, titles, y_lims):
        ax.set_ylim(ylim[0], ylim[1])
        # ax.set_xlim(val_x_axis[0] * 1.02, val_x_axis[-1] * 1.02)
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 10, 10))
        ax.set_xticks(np.linspace(val_x_axis[0], val_x_axis[-1], 7))
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(r"Accuracy(\%)", fontsize=20)
        ax.hlines(np.arange(ylim[0], ylim[1] + 10, 10), val_x_axis[0], val_x_axis[-1], colors="gray", alpha=0.3)
        ax.set_title(ttl, fontsize=25)
        ax.legend()
        if reverse_x_axis:
            ax.set_xlim(val_x_axis[-1] * 1.05, val_x_axis[0] * 0.95)
        else:
            ax.set_xlim(val_x_axis[0] * 0.95, val_x_axis[-1] * 1.05)
    fig.tight_layout()
    plt.show()

def __plot_ablation_uncertainty(id_auroc: np.ndarray, ood_auroc: np.ndarray, val_x_axis: np.ndarray, x_label: str, ylims: tuple,
                                titles: list, reverse_x_axis: bool = False):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(val_x_axis, id_auroc, label="In-Distribution AUROC", color="blue")
    axs[1].plot(val_x_axis, ood_auroc, label="Out-Of-Distribution AUROC", color="red")
    for ax, ttl, ylim in zip(axs, titles, ylims):
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xticks(np.linspace(val_x_axis[0], val_x_axis[-1], 8))
        ax.set_yticks(np.arange(ylim[0], ylim[1], 0.1))
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel("Value", fontsize=20)
        if reverse_x_axis:
            ax.set_xlim(val_x_axis[-1] * 1.02, val_x_axis[0] * 0.95)
            ax.hlines(np.arange(0, 1.1, 0.1), val_x_axis[-1], val_x_axis[0], colors="gray", alpha=0.3)
        else:
            ax.hlines(np.arange(0, 1.1, 0.1), val_x_axis[0], val_x_axis[-1], colors="gray", alpha=0.3)
            ax.set_xlim(val_x_axis[0] * 0.95, val_x_axis[-1] * 1.05)
        ax.legend()
        ax.set_title(ttl, fontsize=25)
    fig.tight_layout()
    plt.show()

def plot_privacy_study(dset_name: str = "MNIST", lf_steps: int = 10, epsilon:int = 15, num_chains: int = 1, tau_g: int = 2, tau_l: int = 2):
    plt.rcParams['text.usetex'] = True
    with open("privacy_study.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        results = config[dset_name]

    epochs_trained = [r["num_epochs"] for r in results]
    deltas = [get_delta_dp_bound(epsilon, num_chains, ne, lf_steps, tau_g, tau_l) for ne in epochs_trained]
    oods, ibps, stds = [r["ood_auroc"] for r in results], [r["ibp_acc"] / 100 for r in results], [r["std_acc"] / 100 for r in results]
    # plot them on 1 single graph (4 lines)
    plt.plot(epochs_trained, deltas, label=r"DP-$\delta$", color="black", linewidth=5)
    plt.plot(epochs_trained, oods, label="OOD AUROC", color="blue", linewidth=3)
    plt.plot(epochs_trained, ibps, label="IBP Accuracy", color="red", linewidth=3)
    if dset_name == "MNIST":
        plt.plot(epochs_trained, stds, label="Standard Accuracy", color="green", linewidth=3)
        plt.title(r"\textbf{MNIST model properties (DP-$\mathbf{\epsilon = 15}$})", fontsize=35)
    else:
        plt.title(r"\textbf{PneumoniaMNIST model properties (DP-$\mathbf{\epsilon = 10}$})", fontsize=35)
        plt.plot(epochs_trained, stds, label="Standard Accuracy", color="green", linewidth=3, alpha=1, linestyle="--")
    plt.ylabel(r"\textbf{Value}", fontsize=28)
    plt.xlabel(r"\textbf{Number of epochs trained}", fontsize=28)
    plt.xticks(epochs_trained, fontsize=15)
    hlines = np.arange(0, 1.1, 0.1)
    plt.hlines(hlines, epochs_trained[0], epochs_trained[-1], colors="gray", alpha=0.3)
    plt.yticks(hlines, fontsize=15)

    plt.legend(prop={'size': 18})
    plt.show()

plot_ablation("PNEUMONIA")
