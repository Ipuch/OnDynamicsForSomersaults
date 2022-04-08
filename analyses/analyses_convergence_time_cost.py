"""
This script is used to plot the convergence time and cost of the different MillerDynamics in implicit form.
when increasing the number of shooting points
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

# select only the one who converged
df_results = df_results[df_results["status"] == 0]
df_results["computation_time"] = df_results["computation_time"] / 60

fig = make_subplots(rows=1, cols=2, subplot_titles=("", ""))

for jj, d in enumerate(dyn):
    fig = my_shaded_trace(fig, df_results, d, pal[jj], grps[jj], key="computation_time", row=1, col=1)

# Update xaxis properties
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=12),
    row=1,
    col=1,
)
# Update yaxis properties
fig.update_yaxes(
    title_text=r"$\text{Convergence time (min)}$",
    showline=True,
    linecolor="black",
    ticks="outside",
    type="log",
    title_standoff=0,
    exponentformat="e",
    row=1,
    col=1,
)

for jj, d in enumerate(dyn):
    fig = my_shaded_trace(fig, df_results, d, pal[jj], grps[jj], key="cost_J", row=1, col=2, show_legend=False)

# Update xaxis properties
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=12),
    row=1,
    col=2,
)
# Update yaxis properties
fig.update_yaxes(
    title_text=r"$\mathcal{J}_1 + \mathcal{J}_2$",
    showline=True,
    linecolor="black",
    ticks="outside",
    type="linear",
    title_standoff=0,
    exponentformat="e",
    row=1,
    col=2,
)

fig = add_annotation_letter(fig, "A", x=0.01, y=0.99, on_paper=True)
fig = add_annotation_letter(fig, "B", x=0.56, y=0.99, on_paper=True)

fig.update_layout(
    height=400,
    width=800,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=10),
        xanchor="right",
        yanchor="top",
        x=1,
        y=1.05,
    ),
    font=dict(
        size=12,
        family="Times New Roman",
    ),
    xaxis=dict(color="black"),
    yaxis=dict(color="black"),
    template="simple_white",
)


fig.show()
fig.write_image(out_path_file + "/analyse_convergence_time_cost.png")
fig.write_image(out_path_file + "/analyse_convergence_time_cost.pdf")
fig.write_image(out_path_file + "/analyse_convergence_time_cost.eps")
fig.write_html(out_path_file + "/analyse_convergence_time_cost.html")
