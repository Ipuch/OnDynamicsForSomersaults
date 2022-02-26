from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

out_path_file = "../../OnDynamicsForSommersaults_results/figures/V1"

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

# sns.lineplot(x="time", y="signal",
#              hue="dynamics",
#              data=df_results)

# get dataframes of shape
# for explicit ... root_implicit_qdddot
# for each q, qdot, qddot, tau
# company
# time  i_rand1  i_rand2 etc...
# 0.1  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
# df = px.data.stocks(indexed=True) - 1
# fig = px.area(df, facet_col="company", facet_col_wrap=2)

print("hey")
dyn = df_results["dynamics_type_label"].unique()
grps = ["Explicit", "Explicit", "Implicit_qddot", "Implicit_qddot", "Implicit_qdddot", "Implicit_qdddot"]
dyn = dyn[[2, 4, 3, 5, 0, 1]]
print(dyn)

fig = make_subplots(rows=1, cols=2)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]
df_results["computation_time"] = df_results["computation_time"] / 60
df_results["iter_per_sec"] = df_results["iterations"] / df_results["computation_time"]


def my_traces(fig, dyn, grps, df, key, row, col, title_str):
    if col > 1 or row > 1:
        showleg = False
    else:
        showleg = True

    for ii, d in enumerate(dyn):
        # manage color
        c = px.colors.hex_to_rgb(px.colors.qualitative.D3[ii])
        c = str(f"rgba({c[0]},{c[1]},{c[2]},0.5)")
        fig.add_trace(
            go.Box(
                x=df["dynamics_type_label"][df["dynamics_type_label"] == d],
                y=df[key][df["dynamics_type_label"] == d],
                name=d,
                boxpoints="all",
                width=0.4,
                pointpos=-2,
                legendgroup=grps[ii],
                fillcolor=c,
                marker=dict(opacity=0.5),
                line=dict(color=px.colors.qualitative.D3[ii]),
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
        title=title_str,
        title_standoff=2,
        domain=[0, 1],
        tickson="boundaries",
        # tick0=2,  # a ne pas garder
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


fig = my_traces(fig, dyn, grps, df_results, "computation_time", 1, 1, r"$\text{time (min)}$")
fig = my_traces(fig, dyn, grps, df_results, "iter_per_sec", 1, 2, r"$\text{iterations / min}$")

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

fig.write_image(out_path_file + "/analyse_time_iter.png")
fig.write_image(out_path_file + "/analyse_time_iter.pdf")
fig.write_html(out_path_file + "/analyse_time_iter.html")
