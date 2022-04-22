"""
This script plots all generalized coordinates and all torques for all MillerDynamics
contained in the main cluster of optimal costs
It requires the dataframe of all results to run the script.
"""

from custom_dynamics.enums import MillerDynamics
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import biorbd

df_results = pd.read_pickle("Dataframe_convergence_metrics_5.pkl")
out_path_file = "../../OnDynamicsForSommersaults_results/figures/V5"

df_results = df_results[df_results["status"] == 0]

n_shooting_tot = df_results["n_shooting_tot"].unique()
for d in MillerDynamics:
    for n in n_shooting_tot:
        df_results = df_results.drop(df_results[(df_results["dynamics_type"] == d) & (df_results["n_shooting_tot"] == n)].index[1:])

df_results["grps"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "grps"] = "Explicit"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "grps"] = "Root_Explicit"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "grps"] = "Implicit_qddot"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "grps"] = "Root_Implicit_qddot"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "grps"] = "Implicit_qdddot"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "grps"] = "Root_Implicit_qdddot"

df_results["dyn_num"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "dyn_num"] = 0
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dyn_num"] = 1
df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dyn_num"] = 2
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dyn_num"] = 3
df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dyn_num"] = 4
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dyn_num"] = 5

grps = ["Explicit", "Root_Explicit", "Implicit_qddot", "Root_Implicit_qddot", "Implicit_qdddot", "Root_Implicit_qdddot"]
df_results["grps"] = pd.Categorical(df_results["grps"], grps)
df_results = df_results.sort_values("grps")

nq = df_results["q_integrated"].iloc[0].shape[0]
rows = 5
cols = 3
idx_rows, idx_cols = np.unravel_index([i for i in range(nq)], (rows, cols))
idx_rows += 1
idx_cols += 1
model_path = df_results["model_path"].iloc[0]
model = biorbd.Model(f"../{model_path}")
list_dof = [dof.to_string() for dof in model.nameDof()]
list_dof_label = [
    "Pelvis X Translation",
    "Pelvis Y Translation",
    "Pelvis Z Translation",
    "Pelvis X Rotation - <i>sommersault</i>",
    "Pelvis Y Rotation - <i>tilt</i>",
    "Pelvis X Rotation - <i>twist</i>",
    "Thoracolumbar Flexion(-)/Extension(+)",
    "Thoracolumbar Lateral Bending Right(+)/Left(-)",
    "Thoracolumbar Axial rotation Left(+)/Right(-)",
    "Right arm Plane of Elevation",
    "Right arm Elevation",
    "Left arm Plane of Elevation",
    "Left arm Elevation",
    "Hips Flexion(+)/Extension(-)",
    "Hips Abduction Right(-)/Left(+)",
]


def plot_all_dof(fig, key: str, df_results, list_dof, idx_rows, idx_cols):
    first_e = 0
    first_re = 0
    first_i = 0
    first_ri = 0
    first_iqdddot = 0
    first_riqdddot = 0
    showleg = False

    for i_dof, dof_name in enumerate(list_dof):
        idx_row = idx_rows[i_dof]
        idx_col = idx_cols[i_dof]
        for index, row in df_results.iterrows():
            showleg = False
            if i_dof == 0:
                if row.dynamics_type == MillerDynamics.EXPLICIT:
                    if first_e == 0:
                        showleg = True
                if row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
                    if first_re == 0:
                        showleg = True
                if row.dynamics_type == MillerDynamics.IMPLICIT:
                    if first_i == 0:
                        showleg = True
                if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
                    if first_ri == 0:
                        showleg = True
                if row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
                    if first_iqdddot == 0:
                        showleg = True
                if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
                    if first_riqdddot == 0:
                        showleg = True

            coef = 180 / np.pi if i_dof > 2 and "q" in key else 1
            c = px.colors.hex_to_rgb(px.colors.qualitative.D3[row.dyn_num])
            color_str = f"rgba({c[0]},{c[1]},{c[2]},{row.n_shooting_tot/840})"
            fig.add_scatter(
                x=row.t_integrated,
                y=row[key][i_dof] * coef,
                mode="lines",
                marker=dict(
                    size=0.2,
                    color=color_str,
                    line=dict(width=3, color=color_str),
                ),
                name=row.dynamics_type_label,
                legendgroup=row.grps,
                showlegend=showleg,
                row=idx_row,
                col=idx_col,
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
        height=1200,
        width=1200,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(family="Times New Roman", color="black", size=15),
            orientation="h",
            yanchor="bottom",
            y=1.05,
            x=0.5,
            xanchor="center",
            valign="top",
        ),
        font=dict(
            size=12,
            family="Times New Roman",
        ),
        yaxis=dict(color="black"),
        template="simple_white",
        boxgap=0.2,
    )
    return fig


fig = make_subplots(rows=rows, cols=cols, subplot_titles=list_dof_label, vertical_spacing=0.05, shared_xaxes=True)
fig = plot_all_dof(fig, "tau_integrated", df_results, list_dof, idx_rows, idx_cols)
fig.update_yaxes(row=1, col=1, title=r"$\boldsymbol{\tau} \; \text{(N)}$")
for i in range(2, rows + 1):
    fig.update_yaxes(row=i, col=1, title=r"$\boldsymbol{\tau} \; \text{(Nm)}$")
for i in range(1, cols + 1):
    fig.update_xaxes(row=rows, col=i, title=r"$\text{Time (s)}$")
fig.show()

fig = make_subplots(rows=rows, cols=cols, subplot_titles=list_dof_label, vertical_spacing=0.05, shared_xaxes=True)
fig = plot_all_dof(fig, "q_integrated", df_results, list_dof, idx_rows, idx_cols)
fig.update_yaxes(row=1, col=1, title=r"$\boldsymbol{q} \; \text{(m)}$")
for i in range(2, rows + 1):
    fig.update_yaxes(row=i, col=1, title=r"$\boldsymbol{q}\; \text{(degree)}$")
for i in range(1, cols + 1):
    fig.update_xaxes(row=rows, col=i, title=r"$\text{Time (s)}$")
fig.show()

fig = make_subplots(rows=rows, cols=cols, subplot_titles=list_dof_label, vertical_spacing=0.05, shared_xaxes=True)
fig = plot_all_dof(fig, "qdot_integrated", df_results, list_dof, idx_rows, idx_cols)
fig.update_yaxes(row=1, col=1, title=r"$\dot{\boldsymbol{q}}\; \text{(m/s)}$")
for i in range(2, rows + 1):
    fig.update_yaxes(row=i, col=1, title=r"$\dot{\boldsymbol{q}}\; \text{(deg/s)}$")
for i in range(1, cols + 1):
    fig.update_xaxes(row=rows, col=i, title=r"$\text{Time (s)}$")
fig.show()

fig = make_subplots(rows=rows, cols=cols, subplot_titles=list_dof_label, vertical_spacing=0.05, shared_xaxes=True)
fig = plot_all_dof(fig, "qddot_integrated", df_results, list_dof, idx_rows, idx_cols)
fig.update_yaxes(row=1, col=1, title=r"$\ddot{\boldsymbol{q}}\; \text{(m/s\superscript{2})}$")
for i in range(2, rows + 1):
    fig.update_yaxes(row=i, col=1, title=r"$\ddot{\boldsymbol{q}}\; \text{(deg/s\superscript{2})}$")
for i in range(1, cols + 1):
    fig.update_xaxes(row=rows, col=i, title=r"$\text{Time (s)}$")
fig.show()

# l'intégration des termes non-piecewise constant est grandement influencée par l'intégration carré dans la fonction de cout
# on observe que les pics sont réduit au fur et à mesure que la fonction converge.
