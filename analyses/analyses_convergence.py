"""
This script is used to plot angular momentum, linear momentum, torque and force residuals for the convergence analysis.
It requires the dataframe of all results to run the script.
"""
from custom_dynamics.enums import MillerDynamics
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from utils import my_shaded_trace, add_annotation_letter

out_path_file = "../../OnDynamicsForSommersaults_results/figures/V5"
model = "../Model_JeCh_15DoFs.bioMod"

df_results = pd.read_pickle("Dataframe_convergence_metrics_5.pkl")

# List of dynamics
dynamic_types = [
    MillerDynamics.IMPLICIT,
    MillerDynamics.ROOT_IMPLICIT,
    MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT,
    MillerDynamics.ROOT_IMPLICIT_QDDDOT,
]

dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[1, 3, 0, 2]]
grps = ["Implicit_qddot", "root_Implicit_qddot", "Implicit_qdddot", "root_Implicit_qdddot"]
pal = px.colors.qualitative.D3[2:]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        r"$\text{Linear Momentum}$",
        r"$\text{Angular Momentum}$",
        r"$\text{Forces}$",
        r"$\text{Torques}$",
    ),
    vertical_spacing=0.1,
    horizontal_spacing=0.15,
)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]
df_results = df_results[df_results["n_shooting_tot"] >= 150]

row_list = [1, 1, 2, 2]
col_list = [1, 2, 1, 2]
show_legend = [True, False, False, False]
keys = ["linear_momentum_rmse", "angular_momentum_rmse", "int_T", "int_R"]
for jj, d in enumerate(dyn):
    for ii, k in enumerate(keys):
        fig = my_shaded_trace(
            fig,
            df_results,
            d,
            pal[jj],
            grps[jj],
            key=k,
            row=row_list[ii],
            col=col_list[ii],
            show_legend=show_legend[ii],
        )

# Update xaxis properties
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    row=2,
    col=1,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=15),
    range=[140, 850],
    title_standoff=0,
)
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    row=2,
    col=2,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=15),
    range=[140, 850],
    title_standoff=0,
)
fig.update_xaxes(
    title_text="",
    row=1,
    col=1,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=15),
    range=[140, 850],
)
fig.update_xaxes(
    title_text="",
    row=1,
    col=2,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=15),
    range=[140, 850],
)

# Update yaxis properties
fig.update_yaxes(
    title_text=r"$\text{Residuals (N.m.s)}$",
    row=2,
    col=2,
    showline=True,
    linecolor="black",
    ticks="outside",
    type="log",
    title_standoff=0,
    exponentformat="e",
)

fig.update_yaxes(
    title_text=r"$\text{Residuals (N.s)}$",
    row=2,
    col=1,
    showline=True,
    linecolor="black",
    ticks="outside",
    type="log",
    title_standoff=0,
    exponentformat="e",
)

fig.update_yaxes(
    title_text=r"$\textrm{RMSe  (}kg.m.s^{-1}\text{)}$",
    row=1,
    col=1,
    showline=True,
    linecolor="black",
    ticks="outside",
    type="log",
    title_standoff=0,
    exponentformat="e",
)
fig.update_yaxes(
    title_text=r"$\textrm{RMSe  (}kg.m^2.s^{-1}\text{)}$",
    row=1,
    col=2,
    showline=True,
    linecolor="black",
    ticks="outside",
    type="log",
    title_standoff=0,
    exponentformat="e",
)

fig.update_layout(
    height=800,
    width=800,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=15),
        orientation="h",
        xanchor="center",
        x=0.5,
    ),
    font=dict(
        size=17,
        family="Times New Roman",
    ),
    xaxis=dict(color="black"),
    yaxis=dict(color="black"),
    template="simple_white",
)
fig = add_annotation_letter(fig, "A", x=0.01, y=0.99, on_paper=True)
fig = add_annotation_letter(fig, "B", x=0.59, y=0.99, on_paper=True)
fig = add_annotation_letter(fig, "C", x=0.01, y=0.44, on_paper=True)
fig = add_annotation_letter(fig, "D", x=0.59, y=0.44, on_paper=True)

fig.show()
fig.write_image(out_path_file + "/analyse_convergence.png")
fig.write_image(out_path_file + "/analyse_convergence.pdf")
fig.write_image(out_path_file + "/analyse_convergence.eps")
fig.write_html(out_path_file + "/analyse_convergence.html")

fig.update_layout(
    height=1200,
    width=1200,
)

fig.show()
fig.write_image(out_path_file + "/analyse_convergence_large.png")
fig.write_image(out_path_file + "/analyse_convergence_large.pdf")
fig.write_image(out_path_file + "/analyse_convergence_large.eps")
fig.write_html(out_path_file + "/analyse_convergence_large.html")
