import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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


def my_traces(fig, dyn, grps, df, key, row, col, title_str):
    if (col == 1 and row == 1) or (col is None or row is None):
        showleg = True
    else:
        showleg = False

    for ii, d in enumerate(dyn):
        # manage color
        c = px.colors.hex_to_rgb(px.colors.qualitative.D3[ii])
        c = str(f"rgba({c[0]},{c[1]},{c[2]},0.5)")
        fig.add_trace(go.Box(x=df["dynamics_type_label"][df["dynamics_type_label"] == d],
                             y=df[key][df["dynamics_type_label"] == d],
                             name=d,
                             boxpoints="all",
                             width=0.4,
                             pointpos=-2,
                             legendgroup=grps[ii],
                             fillcolor=c,
                             marker=dict(opacity=0.5),
                             line=dict(color=px.colors.qualitative.D3[ii])),
                      row=row,
                      col=col,
                      )

    fig.update_traces(
        jitter=0.8,  # add some jitter on points for better visibility
        marker=dict(size=3),
        row=row,
        col=col,
        showlegend=showleg, selector=dict(type='box'),
    )
    fig.update_yaxes(type="log", row=row,
                     col=col, title=title_str,
                     title_standoff=2,
                     domain=[0, 1],
                     tickson="boundaries",
                     # tick0=2,  # a ne pas garder
                     exponentformat='e',
                     ticklabeloverflow="allow"
                     )
    fig.update_xaxes(row=row,
                     col=col, color="black",
                     showticklabels=False,
                     ticks="",
                     )  # no xticks)
    return fig
