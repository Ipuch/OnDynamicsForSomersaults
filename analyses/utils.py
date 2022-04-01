import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats
import math


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def comdot_time_series(m, q, qdot):
    n = q.shape[1]
    comdot = np.zeros((3, n))
    for ii in range(n):
        comdot[:, ii] = m.CoMdot(q[:, ii], qdot[:, ii], True).to_array()

    return comdot


def comddot_time_series(m, q, qdot, qddot):
    n = q.shape[1]
    comddot = np.zeros((3, n))
    for ii in range(n):
        comddot[:, ii] = m.CoMddot(q[:, ii], qdot[:, ii], qddot[:, ii], True).to_array()

    return comddot


def angular_momentum_time_series(m, q, qdot):
    n = q.shape[1]
    angular_momentum = np.zeros((3, n))
    for ii in range(n):
        angular_momentum[:, ii] = m.angularMomentum(q[:, ii], qdot[:, ii], True).to_array()

    return angular_momentum


def residual_torque_time_series(m, q, qdot, qddot):
    n = q.shape[1]
    residual_torque = np.zeros((6, n))
    for ii in range(n):
        residual_torque[:, ii] = m.InverseDynamics(q[:, ii], qdot[:, ii], qddot[:, ii]).to_array()[:6]

    return residual_torque


def linear_momentum_time_series(m, q, qdot):
    n = q.shape[1]
    linear_momentum = np.zeros((3, n))
    for ii in range(n):
        linear_momentum[:, ii] = m.mass() * m.CoMdot(q[:, ii], qdot[:, ii], True).to_array()

    return linear_momentum


def angular_momentum_deviation(angular_momentum):
    n = angular_momentum.shape[1]
    angular_momentum_norm = np.zeros(n)
    for i in range(n):
        angular_momentum_norm[i] = np.linalg.norm(angular_momentum[:, i])

    return rmse(angular_momentum_norm, np.repeat(np.linalg.norm(angular_momentum[:, 0:1]), axis=0, repeats=n))


def linear_momentum_deviation(mass, com_velocity, time, com_acceleration):
    n = com_velocity.shape[1]
    linear_momentum_norm = np.zeros(n)

    com_velocity0 = np.zeros((3, n))
    c0 = com_velocity[0:2, 0]
    com_velocity0[0:2, :] = np.repeat(c0[:, np.newaxis], n, axis=1)
    com_velocity0[2, :] = com_acceleration[2, 0] * time + com_velocity[2, 0]

    for i in range(n):
        linear_momentum_norm[i] = mass * np.linalg.norm(com_velocity[:, i])

    return rmse(linear_momentum_norm, mass * np.linalg.norm(com_velocity0, axis=0))


def stack_states(states, key: str = "q"):
    return np.hstack((states[0][key][:, :-1], states[1][key][:, :]))


def stack_controls(controls, key: str = "tau"):
    return np.hstack((controls[0][key][:, :-1], controls[1][key][:, :]))


def define_time(time, n_shooting):
    time_vector = np.hstack(
        (
            np.linspace(0, float(time[0]) - 1 / n_shooting[0] * float(time[0]), n_shooting[0]),
            np.linspace(float(time[0]), float(time[0]) + float(time[1]), n_shooting[1] + 1),
        )
    )
    return time_vector


def define_integrated_time(time, n_shooting, nstep):
    time_integrated = np.array([])
    cum_time = get_cum_time(time)
    for i, n_shoot in enumerate(n_shooting):
        periode = (cum_time[i + 1] - cum_time[i]) / n_shoot
        for ii in range(n_shoot):
            t_linspace = np.linspace(cum_time[i] + ii * periode, cum_time[i] + (ii + 1) * periode, (nstep + 1))
            time_integrated = np.hstack((time_integrated, t_linspace))
        time_integrated = np.hstack((time_integrated, cum_time[i + 1]))
    return time_integrated


def define_control_integrated(controls, nstep: int, key: str = "tau"):
    tau_integrated = np.hstack(
        (
            np.repeat(controls[0][key], nstep + 1, axis=1)[:, :-nstep],
            np.repeat(controls[1][key], nstep + 1, axis=1)[:, :-nstep],
        )
    )
    return tau_integrated


def get_cum_time(time):
    time = np.hstack((0, np.array(time).squeeze()))
    cum_time = np.cumsum(time)
    return cum_time


def root_explicit_dynamics(m, q, qdot, qddot_joints):
    mass_matrix_nl_effects = m.InverseDynamics(q, qdot, np.hstack((np.zeros((6,)), qddot_joints))).to_array()[:6]
    mass_matrix = m.massMatrix(q).to_array()
    qddot_base = -np.linalg.solve(mass_matrix[:6, :6], np.eye(6)) @ mass_matrix_nl_effects
    return qddot_base


def my_traces(
    fig,
    dyn,
    grps,
    df,
    key,
    row,
    col,
    ylabel: str = None,
    title_str: str = None,
    ylog: bool = True,
    color: list = None,
    show_legend: bool = False,
):
    ylog = "log" if ylog == True else None
    if (col == 1 and row == 1) or (col is None or row is None) or show_legend == True:
        showleg = True
    else:
        showleg = False

    for ii, d in enumerate(dyn):
        # manage color
        c = px.colors.hex_to_rgb(px.colors.qualitative.D3[ii]) if color is None else color[ii]
        c = str(f"rgba({c[0]},{c[1]},{c[2]},0.5)")
        c1 = px.colors.qualitative.D3[ii] if color is None else px.colors.label_rgb(color[ii])
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
                line=dict(color=c1),
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
        type=ylog,
        row=row,
        col=col,
        title=ylabel,
        title_standoff=2,
        exponentformat="e",
        # ticklabeloverflow="allow",
    )
    fig.update_xaxes(
        row=row,
        col=col,
        color="black",
        showticklabels=False,
        ticks="",
    )
    return fig


def my_shaded_trace(fig, df, d, color, grps, key, col=None, row=None, show_legend=True):
    my_boolean = df["dynamics_type_label"] == d

    c_rgb = px.colors.hex_to_rgb(color)
    c_alpha = str(f"rgba({c_rgb[0]},{c_rgb[1]},{c_rgb[2]},0.2)")

    fig.add_scatter(
        x=df[my_boolean]["n_shooting_tot"],
        y=df[my_boolean][key],
        mode="markers",
        marker=dict(
            color=color,
            size=3,
        ),
        name=d,
        legendgroup=grps,
        showlegend=False,
        row=row,
        col=col,
    )

    x_shoot = sorted(df[my_boolean]["n_shooting_tot"].unique())

    fig.add_scatter(
        x=x_shoot,
        y=get_all(df, d, key, "mean"),
        mode="lines",
        marker=dict(color=color, size=8, line=dict(width=0.5, color="DarkSlateGrey")),
        name=d,
        legendgroup=grps,
        row=row,
        col=col,
        showlegend=show_legend,
    )

    y_upper = get_all(df, d, key, "ci_up")
    y_upper = [0 if math.isnan(x) else x for x in y_upper]
    y_lower = get_all(df, d, key, "ci_low")
    y_lower = [0 if math.isnan(x) else x for x in y_lower]

    fig.add_scatter(
        x=x_shoot + x_shoot[::-1],  # x, then x reversed
        y=y_upper + y_lower[::-1],  # upper, then lower reversed
        fill="toself",
        fillcolor=c_alpha,
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=grps,
        row=row,
        col=col,
    )
    return fig


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def fn_ci_up(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m + h


def fn_ci_low(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m - h


def get_all(df, dyn_label, data_key, key: str = "mean"):
    my_bool = df["dynamics_type_label"] == dyn_label
    if key == "mean":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].mean()
            for ii in sorted(df[my_bool]["n_shooting_tot"].unique())
        ]
    if key == "max":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].max()
            - df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median()
            for ii in sorted(df[my_bool]["n_shooting_tot"].unique())
        ]
    if key == "min":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].min()
            - df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median()
            for ii in sorted(df[my_bool]["n_shooting_tot"].unique())
        ]
    if key == "median":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].median()
            for ii in sorted(df[my_bool]["n_shooting_tot"].unique())
        ]
    elif key == "std":
        return [
            df[my_bool & (df["n_shooting_tot"] == ii)][data_key].std()
            for ii in sorted(df[my_bool]["n_shooting_tot"].unique())
        ]
    elif key == "ci_up":
        return [
            fn_ci_up(df[my_bool & (df["n_shooting_tot"] == ii)][data_key])
            for ii in sorted(df[my_bool]["n_shooting_tot"].unique())
        ]
    elif key == "ci_low":
        return [
            fn_ci_low(df[my_bool & (df["n_shooting_tot"] == ii)][data_key])
            for ii in sorted(df[my_bool]["n_shooting_tot"].unique())
        ]


def generate_windows_size(nb: int) -> tuple:
    """
    Defines the number of column and rows of subplots from the number of variables to plot.

    Parameters
    ----------
    nb: int
        Number of variables to plot

    Returns
    -------
    The optimized number of rows and columns
    """

    n_rows = int(round(np.sqrt(nb)))
    return n_rows + 1 if n_rows * n_rows < nb else n_rows, n_rows


def add_annotation_letter(
    fig: go.Figure, letter: str, x: float, y: float, row: int = None, col: int = None, on_paper: bool = False
) -> go.Figure:
    """
    Adds a letter to the plot for scientific articles.

    Parameters
    ----------
    fig: go.Figure
        The figure to annotate
    letter: str
        The letter to add to the plot.
    x: float
        The x coordinate of the letter.
    y: float
        The y coordinate of the letter.
    row: int
        The row of the plot to annotate.
    col: int
        The column of the plot to annotate.
    on_paper: bool
        If True, the annotation will be on the paper instead of the axes
    Returns
    -------
    The figure with the letter added.
    """
    if on_paper:
        xref = "paper"
        yref = "paper"
    else:
        xref = f"x{col}" if row is not None else None
        yref = f"y{row}" if row is not None else None

    fig["layout"]["annotations"] += (
        dict(
            x=x,
            y=y,
            xanchor="left",
            yanchor="bottom",
            text=f"{letter})",
            font=dict(family="Times", size=14, color="black"),
            showarrow=False,
            xref=xref,
            yref=yref,
        ),
    )

    return fig
