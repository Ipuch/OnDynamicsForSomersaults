from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

df_results = pd.read_pickle("Dataframe_results_metrics.pkl")

df_results["dynamics_type_label"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "dynamics_type_label"] = r"$\text{Exp-Full}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dynamics_type_label"
] = r"$\text{Exp-Base}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dynamics_type_label"
] = r"$\text{Imp-Full-}\ddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dynamics_type_label"
] = r"$\text{Imp-Base-}\ddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dynamics_type_label"
] = r"$\text{Imp-Full-}\dddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dynamics_type_label"
] = r"$\text{Imp-Base-}\dddot{q}$"

print("hey")
dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[2, 4, 3, 5, 0, 1]]
dyn = dyn[2:]
grps = ["Explicit", "Explicit", "Implicit_qddot", "Implicit_qddot", "Implicit_qdddot", "Implicit_qdddot"]
grps = grps[2:]
colors = px.colors.qualitative.D3[2:]
print(dyn)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        r"$\textrm{Linear Momentum}$",
        r"$\textrm{Angular Momentum}$",
        r"$\textrm{Translation Torque}$",
        r"$\textrm{Rotation Torque}$",
    ),
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
        # range=[np.log10(min_y * 2.5), np.log10(max_y)],
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
    r"$\text{RMSe (}N.s^{-1} \text{)}$",
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
    r"$\text{RMSe (}N.m.s^{-1} \text{)}$",
    r"$\text{Angular Momentum}$",
)
fig = my_traces(
    fig, dyn, grps, colors, df_results, "int_T", 2, 1, r"$\text{Residuals (N.s)}$", r"$\text{Translation torques}$"
)
fig = my_traces(
    fig, dyn, grps, colors, df_results, "int_R", 2, 2, r"$\text{Residuals (N.m.s)}$", r"$\text{Rotation torques}$"
)

fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
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
    # showlegend=False,
    # violingap=0.1,
    boxgap=0.2,
)
fig.show()
fig.write_html("analyse_momentum.html")

print("hello")
