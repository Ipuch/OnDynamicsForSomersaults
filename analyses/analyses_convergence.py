from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

out_path_file = "../../OnDynamicsForSommersaults_results/figures/V1"
path_file = "../../OnDynamicsForSommersaults_results/raw_convergence_merged"
model = "../Model_JeCh_15DoFs.bioMod"

df_results = pd.read_pickle("Dataframe_convergence_metrics.pkl")

# Add dynamic label to the dataframe.
df_results["dynamics_type_label"] = None
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

# List of dynamics
dynamic_types = [
    MillerDynamics.IMPLICIT,
    MillerDynamics.ROOT_IMPLICIT,
    MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT,
    MillerDynamics.ROOT_IMPLICIT_QDDDOT,
]

dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[2, 3, 0, 1]]
grps = ["Implicit_qddot", "root_Implicit_qddot", "Implicit_qdddot", "root_Implicit_qdddot"]
pal = px.colors.qualitative.D3[2:]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        r"$\text{Translation Torque}$",
        r"$\text{Rotation Torque}$",
        r"$\text{Angular Momentum}$",
        r"$\text{Linear Momentum}$",
    ),
)

# select only the one who converged
df_results = df_results[df_results["status"] == 0]


def get_all(df, dyn_label, data_key, key: str = "mean"):
    my_bool = df["dynamics_type_label"] == dyn_label
    if key == "mean":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].mean() for ii in df[my_bool]["n_shooting_tot"].unique()
        ]
    if key == "max":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].max()
            - df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median()
            for ii in df[my_bool]["n_shooting_tot"].unique()
        ]
    if key == "min":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].min()
            - df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median()
            for ii in df[my_bool]["n_shooting_tot"].unique()
        ]
    if key == "median":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median()
            for ii in df[my_bool]["n_shooting_tot"].unique()
        ]
    elif key == "std":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].std() for ii in df[my_bool]["n_shooting_tot"].unique()
        ]


#
# s = 13
# for jj, d in enumerate(dyn):
#     my_boolean = df_results["dynamics_type_label"] == d
#
#     x_shoot = df_results[my_boolean]["n_shooting_tot"].unique()
#     fig.add_scatter(
#         cliponaxis=True,
#         x=x_shoot,
#         y=get_all(df_results, d, "int_T", "mean"),
#         error_y=dict(
#             array=get_all(df_results, d, "int_T", "max"),
#             symmetric=False,
#             arrayminus=get_all(df_results, d, "int_T", "min"),
#             thickness=5,
#         ),
#         marker=dict(color=pal[jj], size=s),
#         mode="markers",
#         opacity=0.5,
#         marker_line_width=2,
#         legendgroup="group2",
#         legendgrouptitle_text="Mean and Standard deviation",
#         name=d,
#         row=1,
#         col=1,
#     )
#     fig.add_scatter(
#         cliponaxis=True,
#         x=x_shoot,
#         y=get_all(df_results, d, "int_R", "mean"),
#         error_y=dict(
#             array=get_all(df_results, d, "int_R", "max"),
#             symmetric=False,
#             arrayminus=get_all(df_results, d, "int_R", "min"),
#             thickness=5,
#         ),
#         marker=dict(color=pal[jj], size=s),
#         mode="markers",
#         opacity=0.5,
#         marker_line_width=2,
#         legendgroup="group2",
#         showlegend=False,
#         legendgrouptitle_text="Mean and Standard deviation",
#         name=d,
#         row=1,
#         col=2,
#     )
#     fig.add_scatter(
#         cliponaxis=True,
#         x=x_shoot,
#         y=get_all(df_results, d, "angular_momentum_rmse", "mean"),
#         error_y=dict(
#             array=get_all(df_results, d, "angular_momentum_rmse", "max"),
#             symmetric=False,
#             arrayminus=get_all(df_results, d, "angular_momentum_rmse", "min"),
#             thickness=5,
#         ),
#         marker=dict(color=pal[jj], size=s),
#         mode="markers",
#         opacity=0.5,
#         marker_line_width=2,
#         legendgroup="group2",
#         showlegend=False,
#         legendgrouptitle_text="Mean and Standard deviation",
#         name=d,
#         row=2,
#         col=1,
#     )
#
#     fig.add_scatter(
#         cliponaxis=True,
#         x=x_shoot,
#         y=get_all(df_results, d, "linear_momentum_rmse", "mean"),
#         error_y=dict(
#             array=get_all(df_results, d, "linear_momentum_rmse", "max"),
#             symmetric=False,
#             arrayminus=get_all(df_results, d, "linear_momentum_rmse", "min"),
#             thickness=5,
#         ),
#         marker=dict(color=pal[jj], size=s),
#         mode="markers",
#         opacity=0.5,
#         marker_line_width=2,
#         legendgroup="group2",
#         showlegend=False,
#         legendgrouptitle_text="Mean and Standard deviation",
#         name=d,
#         row=2,
#         col=2,
#     )


for jj, d in enumerate(dyn):
    my_boolean = df_results["dynamics_type_label"] == d

    c_rgb = px.colors.hex_to_rgb(pal[jj])
    c_alpha = str(f"rgba({c_rgb[0]},{c_rgb[1]},{c_rgb[2]},0.5)")

    fig.add_scatter(
        x=df_results[my_boolean]["n_shooting_tot"],
        y=df_results[my_boolean]["int_T"],
        mode="markers",
        row=1,
        col=1,
        marker=dict(color=c_alpha, size=8, line=dict(width=0.5, color="DarkSlateGrey")),
        name=d,
        legendgroup=grps[jj],
    )

    fig.add_scatter(
        x=df_results[my_boolean]["n_shooting_tot"],
        y=df_results[my_boolean]["int_R"],
        mode="markers",
        row=1,
        col=2,
        marker=dict(color=c_alpha, size=8, line=dict(width=0.5, color="DarkSlateGrey")),
        name=d,
        showlegend=False,
        legendgroup=grps[jj],
    )

    fig.add_scatter(
        x=df_results[my_boolean]["n_shooting_tot"],
        y=df_results[my_boolean]["angular_momentum_rmse"],
        mode="markers",
        row=2,
        col=1,
        marker=dict(color=c_alpha, size=8, line=dict(width=0.5, color="DarkSlateGrey")),
        name=d,
        showlegend=False,
        legendgroup=grps[jj],
    )

    fig.add_scatter(
        x=df_results[my_boolean]["n_shooting_tot"],
        y=df_results[my_boolean]["linear_momentum_rmse"],
        mode="markers",
        row=2,
        col=2,
        marker=dict(color=c_alpha, size=8, line=dict(width=0.5, color="DarkSlateGrey")),
        name=d,
        legendgrouptitle_text="Simulation outputs",
        showlegend=False,
        legendgroup=grps[jj],
    )

# Update xaxis properties
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    row=1,
    col=1,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=10),
)
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    row=1,
    col=2,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=10),
)
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    row=2,
    col=1,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=10),
)
fig.update_xaxes(
    title_text=r"$\textrm{Mesh point number}$",
    row=2,
    col=2,
    showline=True,
    linecolor="black",
    ticks="outside",
    title_font=dict(size=10),
)

# Update yaxis properties
fig.update_yaxes(
    title_text=r"$\text{Residuals (N.m.s)}$",
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
    title_text=r"$\text{Residuals (N.s)}$",
    row=1,
    col=2,
    showline=True,
    linecolor="black",
    ticks="outside",
    type="log",
    title_standoff=0,
    exponentformat="e",
)

fig.update_yaxes(
    title_text=r"$\textrm{RMSe  (}N.m.s^{-1}\text{)}$",
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
    title_text=r"$\textrm{RMSe  (}N.s^{-1}\text{)}$",
    row=2,
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
        y=-0.1,
    ),
    font=dict(
        size=17,
        family="Times New Roman",
    ),
    xaxis=dict(color="black"),
    yaxis=dict(color="black"),
    template="simple_white",
)

fig.show()
fig.write_image(out_path_file + "/analyse_convergence.png")
fig.write_image(out_path_file + "/analyse_convergence.pdf")
fig.write_html(out_path_file + "/analyse_convergence.html")
