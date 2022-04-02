"""
This script is used to plot the linear momentum, angular momentum and force and torque residuals
for the different MillerDynamics
It requires the dataframe of all results to run the script.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils import add_annotation_letter

out_path_file = "../../OnDynamicsForSommersaults_results/figures/V5"
df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")
dyn = [
    "$\\text{Full-Exp}$",
    "$\\text{Base-Exp}$",
    "$\\text{Full-Imp-}\\ddot{q}$",
    "$\\text{Base-Imp-}\\ddot{q}$",
    "$\\text{Full-Imp-}\\dddot{q}$",
    "$\\text{Base-Imp-}\\dddot{q}$",
]

dyn = dyn[2:]
grps = ["Explicit", "Explicit", "Implicit_qddot", "Implicit_qddot", "Implicit_qdddot", "Implicit_qdddot"]
grps = grps[2:]
colors = px.colors.qualitative.D3[2:]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        r"$\textrm{Linear Momentum}$",
        r"$\textrm{Angular Momentum}$",
        r"$\textrm{Forces}$",
        r"$\textrm{Torques}$",
    ),
    vertical_spacing=0.09,
)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]


def my_traces(fig, dyn, grps, c, df, key, row, col, ylabel, title_str: str = None):
    if col > 1 or row > 1:
        showleg = False
    else:
        showleg = True

    for ii, d in enumerate(dyn):
        # manage color
        c_rgb = px.colors.hex_to_rgb(c[ii])
        c_alpha = str(f"rgba({c_rgb[0]},{c_rgb[1]},{c_rgb[2]},0.5)")
        fig.add_trace(
            go.Box(
                x=df["dynamics_type_label"][df["dynamics_type_label"] == d],
                y=df[key][df["dynamics_type_label"] == d],
                name=d,
                boxpoints="all",
                width=0.4,
                pointpos=-2,
                legendgroup=grps[ii],
                fillcolor=c_alpha,
                marker=dict(opacity=0.5),
                line=dict(color=c[ii]),
            ),
            row=row,
            col=col,
        )

    fig.update_traces(
        jitter=0.8,  # add some jitter on points for better visibility
        marker=dict(size=3),
        row=row,
        col=col,
        showlegend=showleg,
        selector=dict(type="box"),
    )
    fig.update_yaxes(
        type="log",
        row=row,
        col=col,
        title=ylabel,
        title_standoff=2,
        tickson="boundaries",
        exponentformat="e",
        ticklabeloverflow="allow",
    )
    fig.update_xaxes(
        row=row,
        col=col,
        color="black",
        showticklabels=False,
        ticks="",
    )  # no xticks)
    return fig


fig = my_traces(
    fig,
    dyn,
    grps,
    colors,
    df_results,
    "linear_momentum_rmse",
    1,
    1,
    r"$\text{RMSe (}kg.m.s^{-1} \text{)}$",
    r"$\text{Linear Momentum}$",
)
fig = my_traces(
    fig,
    dyn,
    grps,
    colors,
    df_results,
    "angular_momentum_rmse",
    1,
    2,
    r"$\text{RMSe (}kg.m^2.s^{-1}\text{)}$",
    r"$\text{Angular Momentum}$",
)
fig = my_traces(fig, dyn, grps, colors, df_results, "int_T", 2, 1, r"$\text{Residuals (N.s)}$", r"$\text{Forces}$")
fig = my_traces(fig, dyn, grps, colors, df_results, "int_R", 2, 2, r"$\text{Residuals (N.m.s)}$", r"$\text{Torques}$")

fig.update_layout(
    height=800,
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
fig = add_annotation_letter(fig, "C", x=0.01, y=0.44, on_paper=True)
fig = add_annotation_letter(fig, "D", x=0.56, y=0.44, on_paper=True)


fig.show()
# fig.write_image(out_path_file + "/analyse_momentum.png")
# fig.write_image(out_path_file + "/analyse_momentum.pdf")
fig.write_image(out_path_file + "/analyse_momentum.eps")
fig.write_html(out_path_file + "/analyse_momentum.html")
