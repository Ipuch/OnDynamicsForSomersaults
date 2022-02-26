from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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


dyn = df_results["dynamics_type_label"].unique()
grps = ["Explicit", "Explicit", "Implicit_qddot", "Implicit_qddot", "Implicit_qdddot", "Implicit_qdddot"]
dyn = dyn[[2, 4, 3, 5, 0, 1]]
print(dyn)
fig = go.Figure()

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

for ii, d in enumerate(dyn):
    # fig.add_trace(go.Violin(x=df_results["dynamics_type_label"][df_results["dynamics_type_label"] == d],
    #                         y=df_results['computation_time'][df_results["dynamics_type_label"] == d],
    #                         name=d,
    #                         box_visible=True,
    #                         meanline_visible=True,
    #                         side="positive"))
    fig.add_trace(
        go.Box(
            x=df_results["dynamics_type_label"][df_results["dynamics_type_label"] == d],
            y=df_results["computation_time"][df_results["dynamics_type_label"] == d] / 60,
            name=d,
            boxpoints="all",
            width=0.4,
            pointpos=-2,
            legendgroup=grps[ii],
        )
    )

fig.update_traces(
    # points='all',  # show all points
    jitter=0.8,  # add some jitter on points for better visibility
    # scalemode='width',
    marker=dict(size=3),
)  # scale violin plot area with total count)
fig.update_yaxes(type="log")
fig.update_layout(
    # xaxis_title=r'$\text{Transcription}$',
    yaxis_title=r"$\text{time (min)}$",
    height=400,
    width=600,
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
    xaxis=dict(
        color="black",
        showticklabels=False,
        ticks="",  # no xticks
    ),
    yaxis=dict(color="black"),
    template="simple_white",
    # showlegend=False,
    # violingap=0.1,
    boxgap=0.2,
)
fig.show()
fig.show()
fig.write_image(out_path_file + "/time.png")
fig.write_image(out_path_file + "/time.pdf")
fig.write_html(out_path_file + "/time.html")
