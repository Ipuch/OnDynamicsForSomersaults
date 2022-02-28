from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from utils import my_traces
import numpy as np

out_path_file = "../../OnDynamicsForSommersaults_results/figures/V3"
df_results = pd.read_pickle("Dataframe_results_metrics_3.pkl")

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

# four first functions for each phase
df_results["cost_J"] = None
df_results["cost_angular_momentum"] = None

for index, row in df_results.iterrows():
    print(index)
    dc = row.detailed_cost

    # Index of cost functions in details costs
    idx_angular_momentum = [5, 6]
    if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        print("coucou")
    if row.dynamics_type == MillerDynamics.EXPLICIT:
        print("coucou")
    if (
        row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT
        or row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
            or row.dynamics_type == MillerDynamics.IMPLICIT
            or row.dynamics_type == MillerDynamics.ROOT_IMPLICIT
    ):
        idx_J = [
            0,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            1,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            2,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            3,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker foot
            4,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            11,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            12,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            13,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            14,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            15,
        ]  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
    else:
        idx_J = [
            0,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            1,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            2,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            3,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker foot
            4,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            10,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            11,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            12,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            13,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            14,
        ]  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof

    df_results.at[index, "cost_J"] = np.sum([row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_J])

    df_results.at[index, "cost_angular_momentum"] = np.sum(
        [row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_angular_momentum]
    )


dyn = df_results["dynamics_type_label"].unique()
grps = ["Explicit", "Explicit", "Implicit_qddot", "Implicit_qddot", "Implicit_qdddot", "Implicit_qdddot"]
dyn = ['$\\text{Exp-Full}$','$\\text{Exp-Base}$', '$\\text{Imp-Full-}\\ddot{q}$', '$\\text{Imp-Base-}\\ddot{q}$',
       '$\\text{Imp-Full-}\\dddot{q}$',
       '$\\text{Imp-Base-}\\dddot{q}$']

fig = make_subplots(rows=1, cols=2)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

fig = my_traces(fig, dyn, grps, df_results, "cost_J", 1, 1, r"$\mathcal{J}_1 + \mathcal{J}_2$")
fig = my_traces(fig, dyn, grps, df_results, "cost_angular_momentum", 1, 2, r"$\mathcal{M}_1$")

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

fig = go.Figure()

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

fig = my_traces(fig, dyn, grps, df_results, "cost_J", None, None, r"$\mathcal{J}_1 + \mathcal{J}_2$")
# fig = my_traces(fig, dyn, grps, df_results, "cost_angular_momentum", 1, 2, r'$\mathcal{M}_1$')

fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
    height=400,
    width=600,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    legend=dict(
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", color="black", size=11),
        # orientation="h",
        # xanchor="center",
        # x=0.5,
        # y=-0.05,
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
fig.write_image(out_path_file + "/detailed_cost.png")
fig.write_image(out_path_file + "/detailed_cost.pdf")
fig.write_html(out_path_file + "/detailed_cost.html")
