import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from globals import ACCURACY_METRICS, UNCERTAINTY_METRICS


#! 5 different models: dnn_sgd, hmc, fgsm_hmc, pgd_hmc, ibp_hmc
#! 8 metrics: std_acc, fgsm_acc, pgd_acc, ibp_acc, in_distrib_auroc, in_distrib_ece, ood_auroc, ood_ece
#? maybe privacy?
def plot_metrics(dataset_name: str = "MNIST", metric_type = "accuracy", for_adv_comparison: bool = True) -> None:
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
        ax[row, col].bar(positions, height=heights, width=0.5, align='center', color=colors, tick_label=models, alpha=0.9)
        ax[row, col].set_ylim(0, 1)
        ax[row, col].set_yticks(hlines)
        ax[row, col].hlines(hlines, -0.5, len(models) - 1 + 0.5, colors="gray", alpha=0.3)
        ax[row, col].set_title(metric)
        ax[row, col].set_ylabel("Accuracy")
        ax[row, col].set_xlabel("Training method")

    fig.show()
    plt.show()

plot_metrics()
plot_metrics(metric_type="uncertainty")
plot_metrics(for_adv_comparison=False)
plot_metrics(metric_type="uncertainty", for_adv_comparison=False)
