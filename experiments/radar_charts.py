from typing import Dict, List

import matplotlib as mpl
import plotly.graph_objects as go
import torch
from experiment_utils import (get_delta_dp_bound_log, load_ablations,
                              load_results)
from matplotlib.colors import Normalize, to_hex
from plotly.subplots import make_subplots
from scipy.optimize import fsolve

from globals import MODEL_NAMES_ADV_DP

PRIVACY_BOUNDS = {"MNIST": {m_name: 1 for m_name in MODEL_NAMES_ADV_DP},
                  "PNEUMONIA_MNIST": {m_name: 1 for m_name in MODEL_NAMES_ADV_DP}}

#TODO: very specific formula, generalize and explain
PRIVACY_BOUNDS["MNIST"]["HMC-DP"] = torch.log(torch.tensor(641250)) / 50
PRIVACY_BOUNDS["MNIST"]["ADV-DP-HMC (IBP)"] = torch.log(torch.tensor(9.49e+6)) / 50
PRIVACY_BOUNDS["PNEUMONIA_MNIST"]["HMC-DP"] = torch.log(torch.tensor(385110)) / 50
PRIVACY_BOUNDS["PNEUMONIA_MNIST"]["ADV-DP-HMC (IBP)"] = torch.log(torch.tensor(6.153e+5)) / 50

def generate_gradient_colors(num_colors, cmap_name='viridis', reversed: bool = False) -> List[str]:
    if reversed:
        cmap_name += '_r'
    cmap = mpl.colormaps[cmap_name]
    norm = Normalize(vmin=0, vmax=num_colors-1)  # Normalize the number of colors
    colors = [to_hex(cmap(norm(i))) for i in range(num_colors)]  # Generate colors in hex format
    return colors


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
    opacities = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    colors = generate_gradient_colors(len(list(net_stats.keys())), cmap_name='autumn')
    fig = go.Figure()
    abl_rob_categories = ['Accuracy', 'Privacy (Certified) [complement on log scale]', "OOD <br> AUROC", "Robustness (Certified)"]
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
            line=dict(color=colors[i]),
            opacity=opacities[i]
        ))


    fig.update_layout(
      polar={"radialaxis": {"visible": True, "range": [0.2, 0.9], "angle": 15, "tickangle": 15, "linecolor": "blue"},
             "angularaxis": {"tickfont": {"size": 24}}},
      showlegend=True,
      legend={"font": {"size": 22}},
      legend_title_text="Perturbation budget:",
      legend_title_font_size=25,
      title={"text": f"<b>Trustworthiness properties of certified ADV-DP-HMC on {dset_name} upon varying the perturbation budget<b><br>",
             "font": {"size": 30}},
      title_x=0.5,
      margin={"l": 50, "r": 100, "t": 90, "b": 70},
      legend_x=0.8,
      legend_y=0.7,
    )
    fig.update_polars(radialaxis={"range": [0.2, 0.9]})

    fig.show()

def make_radar_chart_ablation_privacy(net_stats: Dict[str, List[float]], dset_name: str) -> None:
    # hardcoded for now,
    if dset_name == "MNIST":
        net_stats[0.05]["DP_EPS"] = 9.49e+6
        net_stats[0.1]["DP_EPS"] = 2405040
        net_stats[0.15]["DP_EPS"] = 1092071
        net_stats[0.2]["DP_EPS"] = 632220
        net_stats[0.25]["DP_EPS"] = 419244
        net_stats[0.3]["DP_EPS"] = 303484
    else:
        net_stats[0.1]["DP_EPS"] = 6.153e+5
        net_stats[0.2]["DP_EPS"] = 163698
        net_stats[0.3]["DP_EPS"] = 79850
        net_stats[0.4]["DP_EPS"] = 50439
        net_stats[0.5]["DP_EPS"] = 36801
        net_stats[0.6]["DP_EPS"] = 29381
    colors = generate_gradient_colors(len(list(net_stats.keys())), cmap_name='nipy_spectral', reversed=False)
    colors = ["#fc2003", "#fcf403", "#fcce03", "#03fcbe", "#5a03fc", "#fc03ca", ]
    print(colors)
    fig = go.Figure()
    abl_priv_categories = ['Accuracy', 'Privacy (Certified) [complement on log scale]', "OOD <br> AUROC", "Robustness (Certified)"]
    opacities = [0.8, 0.69, 0.58, 0.47, 0.36, 0.25]
    for i, (tau_g, stats) in enumerate(net_stats.items()):
        stats_values = []
        stats_values.append(stats["STD_ACC"])
        stats_values.append(1 - torch.log(torch.tensor(stats["DP_EPS"])) / 40)
        stats_values.append(stats["OOD_AUROC"])
        stats_values.append(stats["IBP_ACC"])
        # because high ECE is bad
        fig.add_trace(go.Scatterpolar(
            r=stats_values,
            theta=abl_priv_categories,
            fill='toself',
            name=f"{tau_g}",
            line=dict(color=colors[i]),
            opacity=opacities[i]
        ))


    fig.update_layout(
      polar={"radialaxis": {"visible": True, "range": [0.4, 0.9], "angle": 15, "tickangle": 15, "linecolor": "blue"},
             "angularaxis": {"tickfont": {"size": 24}}},
      showlegend=True,
      legend={"font": {"size": 22}},
      legend_title_text="Gradient sensitivity parameter (tau_g):",
      legend_title_font_size=25,
      title={"text": f"<b>Trustworthiness properties of certified ADV-DP-HMC on {dset_name} upon varying privacy guarantees<b>",
             "font": {"size": 30}},
      title_x=0.5,
      margin={"l": 50, "r": 100, "t": 90, "b": 70},
      legend_x=0.8,
      legend_y=0.7,
    )
    fig.update_polars(radialaxis={"range": [0.4, 0.9]})

    fig.show()

def make_radar_chart_ablation_uncertainty(net_stats: Dict[str, List[float]], dset_name: str) -> None:
    colors = ["#fc2003", "#fcf403", "#fcce03", "#03fcbe", "#5a03fc", "#fc03ca", ]
    fig = go.Figure()
    abl_unc_categories = ['Accuracy', 'Privacy (Certified) [complement on log scale]', "OOD <br> AUROC", "Robustness (Certified)"]
    opacities = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    for i, (prior_std, stats) in enumerate(net_stats.items()):
        stats_values = []
        stats_values.append(stats["STD_ACC"])
        stats_values.append(1 - PRIVACY_BOUNDS[dset_name]['ADV-DP-HMC (IBP)'])
        stats_values.append(stats["OOD_AUROC"])
        stats_values.append(stats["IBP_ACC"])
        # because high ECE is bad
        fig.add_trace(go.Scatterpolar(
            r=stats_values,
            theta=abl_unc_categories,
            fill='toself',
            name=f"{prior_std}",
            line=dict(color=colors[i]),
            opacity=opacities[i]
        ))


    fig.update_layout(
      polar={"radialaxis": {"visible": True, "range": [0.45, 0.9], "angle": 15, "tickangle": 15, "linecolor": "blue"},
             "angularaxis": {"tickfont": {"size": 24}}},
      showlegend=True,
      legend={"font": {"size": 22}},
      legend_title_text="Prior standard deviation (mean=0):",
      legend_title_font_size=25,
      title={"text": f"<b>Trustworthiness properties of certified ADV-DP-HMC on {dset_name} upon varying uncertainty guarantees<b>",
             "font": {"size": 30}},
      title_x=0.5,
      margin={"l": 50, "r": 100, "t": 90, "b": 70},
      legend_x=0.8,
      legend_y=0.7,
    )
    fig.update_polars(radialaxis={"range": [0.45, 0.9]})

    fig.show()

def solver(dset_name: str, tau_g: float) -> float:
    func, initial_guess = None, 0
    if dset_name == "MNIST":
        delta_dp = 0.0014
        func = lambda eps_dp: delta_dp - get_delta_dp_bound_log(eps_dp, 3, 65, 120, 0.05, tau_g=tau_g)
        initial_guess = 9.49e+6
    else:
        delta_dp = 0.0018
        func = lambda eps_dp: delta_dp - get_delta_dp_bound_log(eps_dp, 3, 80, 24, 0.1, tau_g=tau_g)
        initial_guess = 383290

    return fsolve(func, initial_guess)[0]

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
ablation_rob_mnist = load_ablations("MNIST", robustness=True, privacy=False)
ablation_rob_pneum = load_ablations("PNEUMONIA_MNIST", robustness=True, privacy=False)
ablation_priv_mnist = load_ablations("MNIST", robustness=False, privacy=True)
ablation_priv_pneum = load_ablations("PNEUMONIA_MNIST", robustness=False, privacy=True)
ablation_unc_mnist = load_ablations("MNIST", robustness=False, privacy=False)
ablation_unc_pneum = load_ablations("PNEUMONIA_MNIST", robustness=False, privacy=False)
ablation_rob_results_mnist, ablation_rob_results_pneum = {}, {}
ablation_priv_results_mnist, ablation_priv_results_pneum = {}, {}
ablation_unc_results_mnist, ablation_unc_results_pneum = {}, {}
# fill in mnist
for ablation_rob_dict_mnist, ablation_priv_dict_mnist, ablation_unc_dict_mnist in zip(ablation_rob_mnist, ablation_priv_mnist, ablation_unc_mnist):
    # extract robustness data
    eps_mnist = ablation_rob_dict_mnist['value']
    ablation_rob_results_mnist[eps_mnist] = {}
    ablation_rob_results_mnist[eps_mnist]['STD_ACC'] = ablation_rob_dict_mnist['std_acc'] / 100
    ablation_rob_results_mnist[eps_mnist]['IBP_ACC'] = ablation_rob_dict_mnist['ibp_acc'] / 100
    ablation_rob_results_mnist[eps_mnist]['OOD_AUROC'] = ablation_rob_dict_mnist['ood_auroc']
    # extract privacy data
    tau_mnist = ablation_priv_dict_mnist['value']
    ablation_priv_results_mnist[tau_mnist] = {}
    ablation_priv_results_mnist[tau_mnist]['STD_ACC'] = ablation_priv_dict_mnist['std_acc'] / 100
    ablation_priv_results_mnist[tau_mnist]['IBP_ACC'] = ablation_priv_dict_mnist['ibp_acc'] / 100
    ablation_priv_results_mnist[tau_mnist]['OOD_AUROC'] = ablation_priv_dict_mnist['ood_auroc']
    # extract uncertainty data
    prior_std_mnist = ablation_unc_dict_mnist['value']
    ablation_unc_results_mnist[prior_std_mnist] = {}
    ablation_unc_results_mnist[prior_std_mnist]['STD_ACC'] = ablation_unc_dict_mnist['std_acc'] / 100
    ablation_unc_results_mnist[prior_std_mnist]['IBP_ACC'] = ablation_unc_dict_mnist['ibp_acc'] / 100
    ablation_unc_results_mnist[prior_std_mnist]['OOD_AUROC'] = ablation_unc_dict_mnist['ood_auroc']
# fill in pneumonia_mnist
for ablation_rob_dict_pneum, ablation_priv_dict_pneum, ablation_unc_dict_pneum in zip(ablation_rob_pneum, ablation_priv_pneum, ablation_unc_pneum):
    # extract robustness data
    eps_pneum = ablation_rob_dict_pneum['value']
    ablation_rob_results_pneum[eps_pneum] = {}
    ablation_rob_results_pneum[eps_pneum]['STD_ACC'] = ablation_rob_dict_pneum['std_acc'] / 100
    ablation_rob_results_pneum[eps_pneum]['IBP_ACC'] = ablation_rob_dict_pneum['ibp_acc'] / 100
    ablation_rob_results_pneum[eps_pneum]['OOD_AUROC'] = ablation_rob_dict_pneum['ood_auroc']
    # extract privacy data
    tau_pneum = ablation_priv_dict_pneum['value']
    ablation_priv_results_pneum[tau_pneum] = {}
    ablation_priv_results_pneum[tau_pneum]['STD_ACC'] = ablation_priv_dict_pneum['std_acc'] / 100
    ablation_priv_results_pneum[tau_pneum]['IBP_ACC'] = ablation_priv_dict_pneum['ibp_acc'] / 100
    ablation_priv_results_pneum[tau_pneum]['OOD_AUROC'] = ablation_priv_dict_pneum['ood_auroc']
    # extract uncertainty data
    prior_std_pneum = ablation_unc_dict_pneum['value']
    ablation_unc_results_pneum[prior_std_pneum] = {}
    ablation_unc_results_pneum[prior_std_pneum]['STD_ACC'] = ablation_unc_dict_pneum['std_acc'] / 100
    ablation_unc_results_pneum[prior_std_pneum]['IBP_ACC'] = ablation_unc_dict_pneum['ibp_acc'] / 100
    ablation_unc_results_pneum[prior_std_pneum]['OOD_AUROC'] = ablation_unc_dict_pneum['ood_auroc']

#@ ----------------- Type 2 experiments (ablation study robustness) -----------------
make_radar_chart_ablation_robustness(ablation_rob_results_mnist, "MNIST")
make_radar_chart_ablation_robustness(ablation_rob_results_pneum, "PNEUMONIA_MNIST")
#@ ----------------------------------------------------------------------------------


#@ ----------------- Type 3 experiments (ablation study privacy (tau_g)) -----------------
# make_radar_chart_ablation_privacy(ablation_priv_results_mnist, "MNIST")
# make_radar_chart_ablation_privacy(ablation_priv_results_pneum, "PNEUMONIA_MNIST")
#@ ---------------------------------------------------------------------------------------

#@ ----------------- Type 4 experiments (ablation study uncertainty (prior_std)) -----------------
# make_radar_chart_ablation_uncertainty(ablation_unc_results_mnist, "MNIST")
# make_radar_chart_ablation_uncertainty(ablation_unc_results_pneum, "PNEUMONIA_MNIST")
#@ -----------------------------------------------------------------------------------------------