"""
This script is used to plot the value of parameters namely final times of phase 1 and phase 2 for each MillerDynamics
It requires the dataframe of all results to run the script.
"""
import pandas as pd
from plotly.subplots import make_subplots
from utils import my_traces

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

fig = my_traces(fig, dyn, grps, df_results, "T1", 1, 1, r"$T_1 \; \text{(s)}$")
fig = my_traces(fig, dyn, grps, df_results, "T2", 1, 2, r"$T_2 \; \text{(s)}$")

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
    boxgap=0.2,
)

fig.show()

fig.write_image(out_path_file + "/parameters.png")
fig.write_image(out_path_file + "/parameters.pdf")
fig.write_html(out_path_file + "/parameters.html")
