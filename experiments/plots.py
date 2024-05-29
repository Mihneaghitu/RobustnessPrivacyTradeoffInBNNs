import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import yaml

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

plot_metrics_individual(dataset_name="PNEUMONIA_MNIST", for_adv_comparison=True, metric_type="uncertainty")
# plot_metrics()
# plot_metrics(metric_type="uncertainty")
# plot_metrics(for_adv_comparison=False)
# plot_metrics(metric_type="uncertainty", for_adv_comparison=False)
#
# plot_metrics(dataset_name="PNEUMONIA_MNIST")
# plot_metrics(dataset_name="PNEUMONIA_MNIST", metric_type="uncertainty")
# plot_metrics(dataset_name="PNEUMONIA_MNIST", for_adv_comparison=False)
# plot_metrics(dataset_name="PNEUMONIA_MNIST", metric_type="uncertainty", for_adv_comparison=False)
