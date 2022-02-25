from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from utils import my_traces

df_results = pd.read_pickle("Dataframe_results_metrics.pkl")

df_results["dynamics_type_label"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "dynamics_type_label"] = r"$\text{Exp-Full}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dynamics_type_label"] = r"$\text{Exp-Base}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dynamics_type_label"] = r"$\text{Imp-Full-}\ddot{q}$"
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dynamics_type_label"] = r"$\text{Imp-Base-}\ddot{q}$"
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dynamics_type_label"] = r"$\text{Imp-Full-}\dddot{q}$"
df_results.loc[
    df_results[
        "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dynamics_type_label"] = r"$\text{Imp-Base-}\dddot{q}$"

df_results["grps"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "grps"] = "Explicit"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "grps"] = "Root_Explicit"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "grps"] = "Implicit_qddot"
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "grps"] = "Root_Implicit_qddot"
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "grps"] = "Implicit_qdddot"
df_results.loc[
    df_results[
        "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "grps"] = "Root_Implicit_qdddot"

df_results["dyn_num"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "dyn_num"] = 0
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dyn_num"] = 1
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dyn_num"] = 2
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dyn_num"] = 3
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dyn_num"] = 4
df_results.loc[
    df_results[
        "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dyn_num"] = 5

dyn = df_results["dynamics_type_label"].unique()
grps = ["Explicit", "Root_Explicit", "Implicit_qddot", "Root_Implicit_qddot", "Implicit_qdddot", "Root_Implicit_qdddot"]
dyn = dyn[[2, 4, 3, 5, 0, 1]]
print(dyn)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

fig = go.Figure()

q_idx = 8
key = "qddot_integrated"

first_e = 0
first_re = 0
first_i = 0
first_ri = 0
first_iqdddot = 0
first_riqdddot = 0
for index, row in df_results.iterrows():
    name = None
    showleg = False
    if row.dynamics_type == MillerDynamics.EXPLICIT:
        if first_e == 0:
            showleg = True
            name = row.dynamics_type_label
    if row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
        if first_re == 0:
            showleg = True
            name = row.dynamics_type_label
    if row.dynamics_type == MillerDynamics.IMPLICIT:
        if first_i == 0:
            showleg = True
            name = row.dynamics_type_label
    if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
        if first_ri == 0:
            showleg = True
            name = row.dynamics_type_label
    if row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
        if first_iqdddot == 0:
            showleg = True
            name = row.dynamics_type_label
    if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        if first_riqdddot == 0:
            showleg = True
            name = row.dynamics_type_label

    fig.add_scatter(
        x=row.t_integrated,
        y=row[key][q_idx, :],
        mode="lines",
        marker=dict(size=1, color=px.colors.qualitative.D3[row.dyn_num],
                    line=dict(width=0.05,
                              color='DarkSlateGrey')
                    ),
        name=name,
        legendgroup=row.grps,
        showlegend=showleg,
    )

    if row.dynamics_type == MillerDynamics.EXPLICIT:
        first_e = 1
    if row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
        first_re = 1
    if row.dynamics_type == MillerDynamics.IMPLICIT:
        first_i = 1
    if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
        first_ri = 1
    if row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
        first_iqdddot = 1
    if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        first_riqdddot = 1

fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
    height=400,
    width=600,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=11),
        # orientation="h",
        # xanchor="center",
        # x=0.5,
        # y=-0.05,
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
