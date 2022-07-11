"""
This script plots all generalized coordinates and all torques for all MillerDynamics
contained in the main cluster of optimal costs
It requires the dataframe of all results to run the script.
"""

from custom_dynamics.enums import MillerDynamics
import pandas as pd
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import biorbd

df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")
out_path_file = "../../OnDynamicsForSommersaults_results/figures/V5"

df_results = df_results[df_results["status"] == 0]
# only the one that were in the main cluster
df_results = df_results[df_results["main_cluster"] == True]
# only one trial by cluster
for d in MillerDynamics:
    df_results = df_results.drop(df_results[df_results["dynamics_type"] == d].index[1:])

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


def plot_all_dof(fig: go.Figure , key: str, df_results: DataFrame, list_dof: list, idx_rows: list, idx_cols: list):
    """
    This function plots all generalized coordinates and all torques for all MillerDynamics
    contained in the main cluster of optimal costs

    Parameters
    ----------
    fig : go.Figure
        Figure to plot on
    key : str
        Key of the dataframe to plot (q_integrated, qdot_integrated, qddot_integrated, tau_integrated)
    df_results : DataFrame
        Dataframe of all results
    list_dof : list
        List of all dofs
    idx_rows : list
        List of the rows to plot
    idx_cols : list
        List of the columns to plot

    Returns
    -------
    fig : go.Figure
        Figure with all plots
    """
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

            if row.dynamics_type == MillerDynamics.EXPLICIT or row.dynamics_type == MillerDynamics.IMPLICIT or row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
                linestyle = "solid"
            else:
                # linestyle = "dot"
                linestyle = "dashdot"

            # tt = [0]
            # for i in range(0,125):
            #     tt.extend([4+6*i, 5+6*i])
            # t =row.t_integrated[[0,4,5]]
            coef = 180 / np.pi if i_dof > 2 and "q" in key else 1

            # snippet to handle not integrated decision variables (q, qdot, qddot, tau)
            if len(list_dof) == row[key].__len__():
                y = row[key][i_dof] * coef
            else:
                if i_dof <= 5:
                    y = np.zeros(row.t.shape)
                else:
                    print(i_dof, i_dof-6)
                    y = row[key][i_dof - 6] * coef
            y[-1] = np.nan

            fig.add_scatter(
                x=row.t_integrated if "integrated" in key else row.t,
                y=y * coef,
                mode="lines",
                marker=dict(
                    size=0.2,
                    color=px.colors.qualitative.D3[row.dyn_num],
                ),
                line=dict(width=1.5, color=px.colors.qualitative.D3[row.dyn_num], dash=linestyle),
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
fig.write_image(out_path_file + "/tau_integrated.png")
fig.write_image(out_path_file + "/tau_integrated.pdf")
fig.write_html(out_path_file + "/tau_integrated.html", include_mathjax="cdn")
fig.write_image(out_path_file + "/tau_integrated.eps")

# zoom on x-axis between 0.7 and 0.75 for each subplot of the figure on the plotly object fig
for i in range(1, rows + 1):
    for j in range(1, cols + 1):
        fig.update_xaxes(row=i, col=j, range=[0.7, 0.75])
fig.show()
fig.write_image(out_path_file + "/tau_zoom.png")
fig.write_image(out_path_file + "/tau_zoom.pdf")
fig.write_html(out_path_file + "/tau_zoom.html", include_mathjax="cdn")
fig.write_image(out_path_file + "/tau_zoom.eps")


fig = make_subplots(rows=rows, cols=cols, subplot_titles=list_dof_label, vertical_spacing=0.05, shared_xaxes=True)
fig = plot_all_dof(fig, "q_integrated", df_results, list_dof, idx_rows, idx_cols)
fig.update_yaxes(row=1, col=1, title=r"$\boldsymbol{q} \; \text{(m)}$")
for i in range(2, rows + 1):
    fig.update_yaxes(row=i, col=1, title=r"$\boldsymbol{q}\; \text{(degree)}$")
for i in range(1, cols + 1):
    fig.update_xaxes(row=rows, col=i, title=r"$\text{Time (s)}$")
fig.show()
fig.write_image(out_path_file + "/q_integrated.png")
fig.write_image(out_path_file + "/q_integrated.pdf")
fig.write_image(out_path_file + "/q_integrated.eps")
fig.write_html(out_path_file + "/q_integrated.html", include_mathjax="cdn")

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

fig = make_subplots(rows=rows, cols=cols, subplot_titles=list_dof_label, vertical_spacing=0.05, shared_xaxes=True)
fig = plot_all_dof(fig, "tau", df_results, list_dof, idx_rows, idx_cols)
fig.update_yaxes(row=1, col=1, title=r"$\boldsymbol{\tau} \; \text{(N)}$")
for i in range(2, rows + 1):
    fig.update_yaxes(row=i, col=1, title=r"$\boldsymbol{\tau} \; \text{(Nm)}$")
for i in range(1, cols + 1):
    fig.update_xaxes(row=rows, col=i, title=r"$\text{Time (s)}$")
fig.show()
fig.write_image(out_path_file + "/tau.png")
fig.write_image(out_path_file + "/tau.pdf")
fig.write_image(out_path_file + "/tau.eps")
fig.write_html(out_path_file + "/tau.html", include_mathjax="cdn")
