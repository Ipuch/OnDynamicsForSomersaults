import numpy as np


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
    angular_momentum_rmsd = np.zeros(n)
    for i in range(n):
        angular_momentum_rmsd[i] = rmse(angular_momentum[:, i], angular_momentum[:, 0])

    return angular_momentum_rmsd


def linear_momentum_deviation(mass, com_velocity, time, com_acceleration):
    n = com_velocity.shape[1]
    linear_momentum_rmsd = np.zeros(n)

    com_velocity0 = np.zeros((3, n))
    c0 = com_velocity[0:2, 0]
    com_velocity0[0:2, :] = np.repeat(c0[:, np.newaxis], n, axis=1)
    com_velocity0[2, :] = (com_acceleration[2, 0] * time + com_velocity[2, 0])

    for i in range(n):
        linear_momentum_rmsd[i] = mass * rmse(com_velocity[:, i], com_velocity0[:, i])

    return linear_momentum_rmsd


def stack_states(states, key: str = "q"):
    return np.hstack((states[0][key][:, :-1], states[1][key][:, :]))


def stack_controls(controls, key: str = "q"):
    return np.hstack((controls[0][key][:, :-1], controls[1][key][:, :]))


def define_time(time, n_shooting):
    time_vector = np.hstack(
        (
            np.linspace(0, float(time[0]) - 1/n_shooting[0] * float(time[0]), n_shooting[0]),
            np.linspace(float(time[0]), float(time[0]) + float(time[1]), n_shooting[1] + 1),
        )
    )
    return time_vector
