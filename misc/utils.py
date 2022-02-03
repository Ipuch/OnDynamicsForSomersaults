import numpy as np


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def angular_momentum_deviation(angular_momentum):
    angular_momentum_rmsd = np.zeros(3)
    for i in range(3):
        angular_momentum_rmsd[i] = rmse(angular_momentum[i, :], angular_momentum[i, 0])

    return angular_momentum_rmsd


def linear_momentum_deviation(mass, com_velocity, time, com_acceleration):
    linear_momentum_rmsd = np.zeros(3)
    for i in range(3):
        if i == 0 or i == 1:
            linear_momentum_rmsd[i] = mass * rmse(com_velocity[i, :], com_velocity[i, 0])
        else:  # in Z direction
            com_velocity0 = (com_acceleration[i, 0] * time + com_velocity[i, 0])
            linear_momentum_rmsd[i] = mass * rmse(com_velocity[i, :], com_velocity0)

    return linear_momentum_rmsd


def stack_states(states, key: str = "q"):
    return np.hstack((states[0][key][:, :-1], states[1][key][:, :]))


def define_time(time, n_shooting):
    time_vector = np.hstack(
        (
            np.linspace(0, float(time[0]), n_shooting[0]),
            np.linspace(float(time[0]), float(time[0]) + float(time[1]), n_shooting[1]+1),
        )
    )
    return time_vector

