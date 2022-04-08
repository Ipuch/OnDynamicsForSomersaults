"""
This script is used to plot the optimal cost of the different MillerDynamics.
It requires the dataframe of all results to run the script.
"""

from custom_dynamics.enums import MillerDynamics
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from utils import my_traces

out_path_file = "../../OnDynamicsForSommersaults_results/figures/V5"
df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")

dyn = df_results["dynamics_type_label"].unique()
grps = ["Explicit", "Explicit", "Implicit_qddot", "Implicit_qddot", "Implicit_qdddot", "Implicit_qdddot"]
dyn = [
    "$\\text{Full-Exp}$",
    "$\\text{Base-Exp}$",
    "$\\text{Full-Imp-}\\ddot{q}$",
    "$\\text{Base-Imp-}\\ddot{q}$",
    "$\\text{Full-Imp-}\\dddot{q}$",
    "$\\text{Base-Imp-}\\dddot{q}$",
]

fig = make_subplots(rows=1, cols=2)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

fig = my_traces(fig, dyn, grps, df_results, "cost_J", 1, 1, r"$\mathcal{J}_1 + \mathcal{J}_2$")
fig = my_traces(fig, dyn, grps, df_results, "cost_angular_momentum", 1, 2, r"$\mathcal{M}_1$")

fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
    height=400,
    width=800,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=11),
        orientation="h",
        xanchor="center",
        x=0.5,
        y=-0.05,
    ),
    font=dict(
        size=12,
        family="Times New Roman",
    ),
    yaxis=dict(color="black"),
    template="simple_white",
    # showlegend=False,
    # violingap=0.1,
    boxgap=0.2,
)
fig.show()

## fig 2
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.02, shared_xaxes=True)
fig = my_traces(fig, dyn, grps, df_results, "cost_J", 1, 1, None, ylog=False)
temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.IMPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT].index)
# fig.update_yaxes(range=[np.log10(temp["cost_J"].min() * 0.9), np.log10(temp["cost_J"].max() * 1.1)], row=1, col=1)
fig.update_yaxes(range=[temp["cost_J"].min() * 0.95, temp["cost_J"].max() * 1.1], row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)

fig = my_traces(fig, dyn, grps, df_results, "cost_J", 2, 1, None, ylog=False)
temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT].index)
# fig.update_yaxes(range=[np.log10(temp["cost_J"].min() * 0.9), np.log10(temp["cost_J"].max() * 1.1)], row=2, col=1)
fig.update_yaxes(range=[temp["cost_J"].min() * 0.95, temp["cost_J"].max() * 1.1], row=2, col=1)

# fig = my_traces(fig, dyn, grps, df_results, "cost_angular_momentum", 1, 2, r'$\mathcal{M}_1$')
fig.add_annotation(
    x=-0.15,
    y=0.5,
    text=r"$\mathcal{J}_1 + \mathcal{J}_2$",
    font=dict(color="black", size=18),
    textangle=270,
    showarrow=False,
    xref="paper",
    yref="paper",
)
fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
    height=400,
    width=600,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=11),
    ),
    font=dict(
        size=12,
        family="Times New Roman",
    ),
    yaxis=dict(color="black"),
    template="simple_white",
    boxgap=0.2,
)
fig.show()
fig.write_image(out_path_file + "/detailed_cost.png")
fig.write_image(out_path_file + "/detailed_cost.pdf")
fig.write_image(out_path_file + "/detailed_cost.eps")
fig.write_html(out_path_file + "/detailed_cost.html", include_mathjax="cdn")

## fig 2
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1)
color = [px.colors.hex_to_rgb(c) for c in px.colors.qualitative.D3[0:6]]
temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.IMPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT].index)
fig = my_traces(fig, dyn[0:2], grps[0:2], temp, "cost_J", 1, 1, None, ylog=False, color=color[0:2], show_legend=True)

temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT].index)
fig = my_traces(fig, dyn[2:4], grps[2:4], temp, "cost_J", 1, 2, None, ylog=False, color=color[2:4], show_legend=True)

temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.IMPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT].index)
fig = my_traces(fig, dyn[4:6], grps[4:6], temp, "cost_J", 1, 3, None, ylog=False, color=color[4:6], show_legend=True)

fig.add_annotation(
    x=-0.15,
    y=0.5,
    text=r"$\mathcal{J}_1 + \mathcal{J}_2$",
    font=dict(color="black", size=18),
    textangle=270,
    showarrow=False,
    xref="paper",
    yref="paper",
)
fig.update_layout(
    height=400,
    width=600,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=11),
    ),
    font=dict(
        size=12,
        family="Times New Roman",
    ),
    yaxis=dict(color="black"),
    template="simple_white",
    boxgap=0.2,
)
fig.show()

label_J = [
    r"$\omega_2 \int_{T_{i-1}}^{T_i} \Delta \dot{\mathbf{q}}^\top_J \Delta \dot{\mathbf{q}}_J\: dt$",
    r"$\omega_1 \; \int_{\widehat{P_{\text{right_hand}}(\mathbf{T})}}  ds$",
    r"$\omega_1 \; \int_{\widehat{P_{\text{left_hand}}(\mathbf{T})}}   ds$",
    r"$\omega_1 \; \int_{\widehat{P_{\text{feet}}(\mathbf{T})}}   ds$",
    r"$\omega_3 \sum_{k \in \mathcal{C}_{dof}} \int_{T_{i-1}}^{T_i} {q_k}^2  \: dt$",
    r"$\omega_2 \int_{T_{i-1}}^{T_i} \Delta \dot{q}^\top_J \Delta \dot{q}_J\: dt$",
    r"$\omega_1 \; \int_{\widehat{P_{\text{right_hand}}(\mathbf{T})}}  ds$",
    r"$\omega_1 \; \int_{\widehat{P_{\text{left_hand}}(\mathbf{T})}}   ds$",
    r"$\omega_1 \; \int_{\widehat{P_{\text{feet}}(\mathbf{T})}}   ds$",
    r"$\omega_3 \sum_{k \in \mathcal{C}_{dof}} \int_{T_{i-1}}^{T_i} {q_k}^2  \: dt$",
]

fig = make_subplots(
    rows=2,
    cols=5,
    subplot_titles=(
        label_J[1],
        label_J[2],
        label_J[3],
        label_J[0],
        label_J[4],
        None,
        None,
        None,
        None,
        None,
    ),
    vertical_spacing=0.05,
)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

ii = 1
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{1}", ii, 1, "Phase 1")
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{2}", ii, 2)
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{3}", ii, 3)
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{0}", ii, 4)
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{4}", ii, 5)
ii = 2
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{6}", ii, 1, "Phase 2")
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{7}", ii, 2)
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{8}", ii, 3)
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{5}", ii, 4)
fig = my_traces(fig, dyn, grps, df_results, f"cost_J{9}", ii, 5)

fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
    height=600,
    width=1200,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=11),
        orientation="h",
        xanchor="center",
        x=0.5,
        y=-0.05,
    ),
    font=dict(
        size=15,
        family="Times New Roman",
    ),
    yaxis=dict(color="black"),
    template="simple_white",
    boxgap=0.2,
)
fig.show()
fig.write_image(out_path_file + "/detailed_detailed_cost.png")
fig.write_image(out_path_file + "/detailed_detailed_cost.pdf")
fig.write_html(out_path_file + "/detailed_detailed_cost.html", include_mathjax="cdn")
fig.write_image(out_path_file + "/detailed_detailed_cost.eps")

# only the one that were in the main cluster
df_results = df_results[df_results["main_cluster"] == True]
## fig 2
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1)
color = [px.colors.hex_to_rgb(c) for c in px.colors.qualitative.D3[0:6]]
temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.IMPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT].index)
fig = my_traces(fig, dyn[0:2], grps[0:2], temp, "cost_J", 1, 1, None, ylog=False, color=color[0:2], show_legend=True)

temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT].index)
fig = my_traces(fig, dyn[2:4], grps[2:4], temp, "cost_J", 1, 2, None, ylog=False, color=color[2:4], show_legend=True)

temp = df_results.drop(df_results[df_results["dynamics_type"] == MillerDynamics.EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.IMPLICIT].index)
temp = temp.drop(temp[temp["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT].index)
fig = my_traces(fig, dyn[4:6], grps[4:6], temp, "cost_J", 1, 3, None, ylog=False, color=color[4:6], show_legend=True)

fig.add_annotation(
    x=-0.15,
    y=0.5,
    text=r"$\mathcal{J}_1 + \mathcal{J}_2$",
    font=dict(color="black", size=18),
    textangle=270,
    showarrow=False,
    xref="paper",
    yref="paper",
)
fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
    height=400,
    width=600,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=11),
    ),
    font=dict(
        size=12,
        family="Times New Roman",
    ),
    yaxis=dict(color="black"),
    template="simple_white",
    boxgap=0.2,
)
fig.show()
