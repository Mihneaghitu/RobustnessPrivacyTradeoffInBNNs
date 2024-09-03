from copy import deepcopy
from typing import Dict, List

import plotly.graph_objects as go
import torch
from experiment_utils import load_ablations, load_results
from plotly.subplots import make_subplots

from globals import MODEL_NAMES_ADV_DP

PRIVACY_BOUNDS = {"MNIST": {m_name: 1 for m_name in MODEL_NAMES_ADV_DP},
                  "PNEUMONIA_MNIST": {m_name: 1 for m_name in MODEL_NAMES_ADV_DP}}

#TODO: very specific formula, generalize and explain
PRIVACY_BOUNDS["MNIST"]["HMC-DP"] = torch.log(torch.tensor(641250)) / 50
PRIVACY_BOUNDS["MNIST"]["ADV-DP-HMC (IBP)"] = torch.log(torch.tensor(9.49e+6)) / 50
PRIVACY_BOUNDS["PNEUMONIA_MNIST"]["HMC-DP"] = torch.log(torch.tensor(613e+3)) / 50
PRIVACY_BOUNDS["PNEUMONIA_MNIST"]["ADV-DP-HMC (IBP)"] = torch.log(torch.tensor(383290)) / 50


def make_radar_chart_performance(net_stats: Dict[str, List[float]], dset_name: str) -> None:
    fig = make_subplots(rows=2, cols=3, start_cell="top-left",
                        specs=[[{"type": "polar"}, {"type": "polar"}, {"rowspan": 2, "type": "polar"}],
                               [{"type": "polar"}, {"type": "polar"}, None]],
                        horizontal_spacing = 0.05,
                        # subplot_titles=["SGD", "HMC-DP", "ADV-DP-HMC (IBP)", "HMC", "ADV-HMC (IBP)"]
                        )
    positions = {0: (1, 1), 1: (1, 2), 2: (2, 1), 3: (2, 2), 4: (1, 3)}
    categories = ['Accuracy',
                  "Robustness <br> (Certified)",
                  "Robustness <br> (FGSM)",
                  "Robustness <br> (PGD)",
                  'Privacy (Certified) <br> [complement on log scale] ',
                  #TODO: Implement MIA (maybe using [LiRA](https://arxiv.org/pdf/2112.03570)
                  # 'Privacy (Membership Inference Attacks)',
                  'OOD AUROC',
                  # because high ECE is bad
                  'OOD ECE <br> [complement]'
                  ]
    for i, (net_name, stats) in enumerate(net_stats.items()):
        stats_values = []
        stats_values.append(stats["STD_ACC"])
        stats_values.append(stats["IBP_ACC"])
        stats_values.append(stats["FGSM_ACC"])
        stats_values.append(stats["PGD_ACC"])
        stats_values.append(1 - PRIVACY_BOUNDS[dset_name][net_name])
        stats_values.append(stats["OOD_AUROC"])
        stats_values.append(1 - stats["OOD_ECE"])
        # because high ECE is bad
        fig.add_trace(go.Scatterpolar(
            r=stats_values,
            theta=categories,
            fill='toself',
            name=f"{net_name}",
            legendgrouptitle=dict(text="Training Methods:", font=dict(size=20)),
        ), row=positions[i][0], col=positions[i][1])


    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True,
      legend=dict(font=dict(size=20)),
      title=dict(text=f"<b>Trustworthiness properties of different algorithms on {dset_name}<b>", font=dict(size=28)),
      title_x=0.5,
    )
    fig.update_polars(radialaxis=dict(range=[0, 1]))

    fig.show()

def make_radar_chart_ablation_robustness(net_stats: Dict[str, List[float]], dset_name: str) -> None:
    fig = make_subplots(rows=2, cols=3, start_cell="top-left",
                        # specs=[[{"type": "polar"} for _ in range(4)], [{"type": "polar"} for _ in range(4)]],
                        specs=[[{"type": "polar"}, {"type": "polar"}, {"type": "polar"}],
                                [{"type": "polar"}, {"type": "polar"}, {"type": "polar"}]],
                        horizontal_spacing = 0.04
                        )
    abl_rob_categories = ['Accuracy', 'Privacy (Certified) <br> [complement on log scale]', "OOD_AUROC", "Robustness <br> (Certified)"]
    for i, (eps_budget, stats) in enumerate(net_stats.items()):
        stats_values = []
        stats_values.append(stats["STD_ACC"])
        stats_values.append(1 - PRIVACY_BOUNDS[dset_name]['ADV-DP-HMC (IBP)'])
        stats_values.append(stats["OOD_AUROC"])
        stats_values.append(stats["IBP_ACC"])
        # because high ECE is bad
        fig.add_trace(go.Scatterpolar(
            r=stats_values,
            theta=abl_rob_categories,
            fill='toself',
            name=f"{eps_budget}",
            legendgrouptitle={"text": r"$\epsilon$", "font": {"size": 35}},
        ), row=i//3+1, col=i%3+1)


    fig.update_layout(
      polar={"radialaxis": {"visible": True, "range": [0, 1]}},
      showlegend=True,
      legend={"font": {"size": 22}},
      title={"text": f"<b>Trustworthiness properties of certified ADV-DP-HMC on {dset_name} upon varying the perturbation budget<b>",
             "font": {"size": 30}},
      title_x=0.5,
      margin={"l": 50, "r": 100, "t": 90, "b": 70},
    )
    fig.update_polars(radialaxis={"range": [0, 1]})

    fig.show()

# Load the yaml files with all the results
results_with_dp_mnist, results_with_dp_pneum = {}, {}
for model_name in MODEL_NAMES_ADV_DP:
    results_with_dp_mnist[model_name] = load_results("MNIST", model_name)
    results_with_dp_pneum[model_name] = load_results("PNEUMONIA_MNIST", model_name)

#@ ----------------- Type 1 experiments (best performance) -----------------
# make_radar_chart_performance(results_with_dp_mnist, "MNIST")
# make_radar_chart_performance(results_with_dp_pneum, "PNEUMONIA_MNIST")
#@ -------------------------------------------------------------------------

# Load ablation yaml files (robustness and privacy)
ablation_rob_mnist = load_ablations("MNIST", robustness=True)
ablation_rob_pneum = load_ablations("PNEUMONIA_MNIST", robustness=True)
ablation_priv_mnist = load_ablations("MNIST", robustness=False)
ablation_priv_pneum = load_ablations("PNEUMONIA_MNIST", robustness=False)
ablation_rob_results_mnist, ablation_rob_results_pneum = {}, {}
ablation_priv_results_mnist, ablation_priv_results_pneum = {}, {}
# fill in mnist
for ablation__rob_dict_mnist, ablation_priv_dict_mnist in zip(ablation_rob_mnist, ablation_priv_mnist):
    eps_mnist = ablation__rob_dict_mnist['value']
    if eps_mnist not in [0.075, 0.1, 0.2, 0.3, 0.4, 0.5]:
        continue
    ablation_rob_results_mnist[eps_mnist] = {}
    ablation_rob_results_mnist[eps_mnist]['STD_ACC'] = ablation__rob_dict_mnist['std_acc'] / 100
    ablation_rob_results_mnist[eps_mnist]['IBP_ACC'] = ablation__rob_dict_mnist['ibp_acc'] / 100
    ablation_rob_results_mnist[eps_mnist]['OOD_AUROC'] = ablation__rob_dict_mnist['ood_auroc']
    dp_mnist = ablation_priv_dict_mnist['value']
    if dp_mnist not in [1, 2, 3, 4, 5, 6]:
        continue
    ablation_priv_results_mnist[dp_mnist] = {}
    ablation_priv_results_mnist[dp_mnist]['STD_ACC'] = ablation_priv_dict_mnist['std_acc'] / 100
    ablation_priv_results_mnist[dp_mnist]['IBP_ACC'] = ablation_priv_dict_mnist['ibp_acc'] / 100
    ablation_priv_results_mnist[dp_mnist]['OOD_AUROC'] = ablation_priv_dict_mnist['ood_auroc']
# fill in pneumonia_mnist
for ablation_dict_pneum in ablation_rob_pneum:
    eps_pneum = ablation_dict_pneum['value']
    if eps_pneum not in [0.01, 0.05, 0.09, 0.13, 0.17, 0.21]:
        continue
    ablation_rob_results_pneum[eps_pneum] = {}
    ablation_rob_results_pneum[eps_pneum]['STD_ACC'] = ablation_dict_pneum['std_acc'] / 100
    ablation_rob_results_pneum[eps_pneum]['IBP_ACC'] = ablation_dict_pneum['ibp_acc'] / 100
    ablation_rob_results_pneum[eps_pneum]['OOD_AUROC'] = ablation_dict_pneum['ood_auroc']
#@ ----------------- Type 2 experiments (ablation study robustness) -----------------
make_radar_chart_ablation_robustness(ablation_rob_results_mnist, "MNIST")
make_radar_chart_ablation_robustness(ablation_rob_results_pneum, "PNEUMONIA_MNIST")
#@ ----------------------------------------------------------------------------------
