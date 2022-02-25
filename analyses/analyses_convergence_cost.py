from custom_dynamics.enums import MillerDynamics
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import seaborn as sns
import scipy.stats

path_file = "../../OnDynamicsForSommersaults_results/raw_convergence_merged"
model = "../Model_JeCh_15DoFs.bioMod"

df_results = pd.read_pickle("Dataframe_convergence_metrics.pkl")

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

# four first functions for each phase
df_results["cost_J"] = None
df_results["cost_angular_momentum"] = None

for index, row in df_results.iterrows():
    print(index)
    dc = row.detailed_cost

    # Index of cost functions in details costs
    idx_angular_momentum = [5, 6]
    if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT \
            or row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
        idx_J = [0,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
                 1,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 2,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 3,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker foot
                 4,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
                 11,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
                 12,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 13,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 14,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
                 15]  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
    else:
        idx_J = [0,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
                 1,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 2,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 3,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker foot
                 4,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
                 10,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
                 11,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 12,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
                 13,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
                 14]  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof

    df_results.at[index, "cost_J"] = \
        np.sum([row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_J])

    df_results.at[index, "cost_angular_momentum"] = \
        np.sum([row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_angular_momentum])

# List of dynamics
dynamic_types = [MillerDynamics.IMPLICIT,
                 MillerDynamics.ROOT_IMPLICIT,
                 MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT,
                 MillerDynamics.ROOT_IMPLICIT_QDDDOT]

dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[2, 3, 0, 1]]
grps = ["Implicit_qddot", "root_Implicit_qddot", "Implicit_qdddot", "root_Implicit_qdddot"]
pal = px.colors.qualitative.D3[2:]

# select only the one who converged
df_results = df_results[df_results["status"] == 0]


fig = go.Figure()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def fn_ci_up(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m + h


def fn_ci_low(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h


def get_all(df, dyn_label, data_key, key: str = "mean"):
    my_bool = df["dynamics_type_label"] == dyn_label
    if key == "mean":
        return [df[my_bool & (df["n_shooting_tot"] == ii)][data_key].mean() for ii in
                sorted(df[my_bool]["n_shooting_tot"].unique())]
    if key == "max":
        return [df[my_bool & (df["n_shooting_tot"] == ii)][data_key].max() -
                df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median() for ii in
                sorted(df[my_bool]["n_shooting_tot"].unique())]
    if key == "min":
        return [df[my_bool & (df["n_shooting_tot"] == ii)][data_key].min()
                - df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median()
                for ii in
                sorted(df[my_bool]["n_shooting_tot"].unique())]
    if key == "median":
        return [df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median() for ii in
                sorted(df[my_bool]["n_shooting_tot"].unique())]
    elif key == "std":
        return [df[my_bool & (df["n_shooting_tot"] == ii)][data_key].std() for ii in
                sorted(df[my_bool]["n_shooting_tot"].unique())]
    elif key == "ci_up":
        return [fn_ci_up(df[my_bool & (df["n_shooting_tot"] == ii)][data_key]) for ii in
                sorted(df[my_bool]["n_shooting_tot"].unique())]
    elif key == "ci_low":
        return [fn_ci_low(df[my_bool & (df["n_shooting_tot"] == ii)][data_key]) for ii in
                sorted(df[my_bool]["n_shooting_tot"].unique())]


s = 13
for jj, d in enumerate(dyn):
    my_boolean = df_results["dynamics_type_label"] == d

    c_rgb = px.colors.hex_to_rgb(pal[jj])
    c_alpha = str(f"rgba({c_rgb[0]},{c_rgb[1]},{c_rgb[2]},0.2)")

    fig.add_scatter(
        x=df_results[my_boolean]["n_shooting_tot"],
        y=df_results[my_boolean]["cost_J"],
        mode="markers",
        marker=dict(color=pal[jj], size=3,
                    # line=dict(width=0.5,
                    #           color='DarkSlateGrey')
                    ),
        name=d,
        legendgroup=grps[jj],
        showlegend=False
    )

    x_shoot = sorted(df_results[my_boolean]["n_shooting_tot"].unique())

    fig.add_scatter(
        x=x_shoot,
        y=get_all(df_results, d, "cost_J", "mean"),
        mode="lines",
        marker=dict(color=pal[jj], size=8,
                    line=dict(width=0.5,
                              color='DarkSlateGrey')
                    ),
        name=d,
        legendgroup=grps[jj]
    )

    y_upper = get_all(df_results, d, "cost_J", "ci_up")
    y_lower = get_all(df_results, d, "cost_J", "ci_low")
    fig.add_scatter(
        x=x_shoot + x_shoot[::-1],  # x, then x reversed
        y=y_upper + y_lower[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor=c_alpha,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=grps[jj],
    )

# Update xaxis properties
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$", showline=True, linecolor="black", ticks="outside",
    title_font=dict(size=10),
)
# Update yaxis properties
fig.update_yaxes(
    title_text=r"$\mathcal{J}_1 + \mathcal{J}_2$",
    showline=True,
    linecolor="black",
    ticks="outside",
    type="linear",
    title_standoff=0,
    exponentformat='e',
)

fig.update_layout(
    height=400,
    width=600,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=12),
        # orientation="v",
        # xanchor="right",
        # x=1.5,
        # y=-.1,
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
# fig.write_html("analyse_convergence_time.html")
