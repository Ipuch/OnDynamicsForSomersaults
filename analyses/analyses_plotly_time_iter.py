"""
This script is used to plot the CPU time and number of iterations of for each OCP of each dynamics type
"""
import pandas as pd
from plotly.subplots import make_subplots
from utils import my_traces, add_annotation_letter

out_path_file = "../../OnDynamicsForSommersaults_results/figures/V5"
df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")

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
df_results["computation_time"] = df_results["computation_time"] / 60
df_results["iter_per_sec"] = df_results["iterations"] / df_results["computation_time"]

fig = my_traces(fig, dyn, grps, df_results, "computation_time", 1, 1, r"$\text{time (min)}$")
fig = my_traces(fig, dyn, grps, df_results, "iterations", 1, 2, r"$\text{iterations}$")

fig.update_layout(
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
    boxgap=0.2,
)
fig = add_annotation_letter(fig, "A", x=0.01, y=0.99, on_paper=True)
fig = add_annotation_letter(fig, "B", x=0.56, y=0.99, on_paper=True)

fig.show()

fig.write_image(out_path_file + "/analyse_time_iter.png")
fig.write_image(out_path_file + "/analyse_time_iter.pdf")
fig.write_image(out_path_file + "/analyse_time_iter.svg")
fig.write_image(out_path_file + "/analyse_time_iter.eps")
fig.write_html(out_path_file + "/analyse_time_iter.html")
