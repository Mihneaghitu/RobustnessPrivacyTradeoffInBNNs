from typing import Dict, List

import plotly.graph_objects as go
from experiment_utils import load_results
from plotly.subplots import make_subplots

from globals import MODEL_NAMES_ADV, MODEL_NAMES_ADV_DP


def make_radar_chart(net_stats: Dict[str, List[float]]) -> None:
    categories = ['Accuracy',
                  "Robustness <br> (Certified)",
                  "Robustness <br> (FGSM)",
                  "Robustness <br> (PGD)",
                  #! Forget about the next 2 for now,
                  #TODO: Figure out a way to get the value of epsilon when the function value of the dp bound does not fit in 64 bits
                  # 'Privacy (Certified)',
                  #TODO: Implement MIA (maybe using [LiRA](https://arxiv.org/pdf/2112.03570)
                  # 'Privacy (Membership Inference Attacks)',
                  'OOD AUROC',
                  # because high ECE is bad
                  'INVERSE OOD ECE']


    fig = make_subplots(rows=3, cols=2, start_cell="top-left",
                        specs=[[{"type": "polar"} for _ in range(2)], [{"type": "polar"} for _ in range(2)],
                               [{"colspan": 2, "type": "polar"}, None]]
                        #subplot_titles=[f"{net_name} \n\n" for net_name in net_stats.keys()]
                        )
    for i, (net_name, stats) in enumerate(net_stats.items()):
        stats_values = []
        stats_values.append(stats["STD_ACC"])
        stats_values.append(stats["IBP_ACC"])
        stats_values.append(stats["FGSM_ACC"])
        stats_values.append(stats["PGD_ACC"])
        stats_values.append(stats["OOD_AUROC"])
        # because high ECE is bad
        stats_values.append(1 - stats["OOD_ECE"])
        fig.add_trace(go.Scatterpolar(
            r=stats_values,
            theta=categories,
            fill='toself',
            name=f"{net_name}",
            legendgrouptitle=dict(text="Training Methods:"),
        ), row=i//2+1, col=i%2+1)


    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True,
      title="Trustworthiness properties of different algorithms on MNIST",
    )
    fig.update_polars(radialaxis=dict(range=[0, 1]))

    fig.show()

results_with_dp_mnist = dict()
for model_name in MODEL_NAMES_ADV_DP:
    results_with_dp_mnist[model_name] = load_results("MNIST", model_name)

make_radar_chart(results_with_dp_mnist)
