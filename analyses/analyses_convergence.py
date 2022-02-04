import os
import pickle
from utils import (stack_states,
                   stack_controls,
                   define_time,
                   angular_momentum_deviation,
                   angular_momentum_time_series,
                   linear_momentum_time_series,
                   linear_momentum_deviation,
                   comdot_time_series,
                   comddot_time_series,
                   residual_torque_time_series)
import biorbd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np

path_file = "../../OnDynamicsForSommersaults_results/raw_converge_analysis"
model = "../Model_JeCh_15DoFs.bioMod"
# ouvrir les fichiers
files = os.listdir(path_file)
files.sort()

colors = pl.cm.jet(np.linspace(0, 1, len(files)))

for i, file in enumerate(files):
    file_path = open(f"{path_file}/{file}", "rb")
    data = pickle.load(file_path)
    print(file + "\n")
    print(data["status"])
    if data["status"] == 0:
        q = stack_states(data["states"], "q")
        qdot = stack_states(data["states"], "qdot")
        qddot = stack_controls(data["controls"], "qddot")
        t = define_time(data["parameters"]["time"], data["n_shooting"])

        m = biorbd.Model(model)

        residual_torque_norm = np.linalg.norm(residual_torque_time_series(m, q, qdot, qddot),axis=0)
        for ii in range(1):
            if ii == 0:
                plt.plot(t, residual_torque_norm, label=data["n_shooting"], color=colors[i])
            else:
                plt.plot(t, residual_torque_norm[ii], color=colors[i])

plt.ylabel("Residual torques norm xyz Rx Ry Rz (N or Nm)")
plt.xlabel("time")
plt.legend()
plt.yscale('log')
plt.show()

for i, file in enumerate(files):
    file_path = open(f"{path_file}/{file}", "rb")
    data = pickle.load(file_path)
    print(file + "\n")
    print(data["status"])

    q = stack_states(data["states"], "q")
    qdot = stack_states(data["states"], "qdot")
    t = define_time(data["parameters"]["time"], data["n_shooting"])

    m = biorbd.Model(model)

    angular_momentum = angular_momentum_time_series(m, q, qdot)
    angular_momentum_rmsd = angular_momentum_deviation(angular_momentum)

    plt.plot(t, angular_momentum_rmsd, label=data["n_shooting"], color=colors[i])
    plt.ylabel("Angular momentum RMSD (kg⋅m2⋅s−1)")
    plt.xlabel("time")
    plt.legend()

plt.show()

for i, file in enumerate(files):
    file_path = open(f"{path_file}/{file}", "rb")
    data = pickle.load(file_path)
    print(file + "\n")
    print(data["status"])

    q = stack_states(data["states"], "q")
    qdot = stack_states(data["states"], "qdot")
    t = define_time(data["parameters"]["time"], data["n_shooting"])

    m = biorbd.Model(model)

    angular_momentum = angular_momentum_time_series(m, q, qdot)

    plt.plot(t, angular_momentum[0], label=data["n_shooting"], color=colors[i])
    plt.plot(t, angular_momentum[1], color=colors[i])
    plt.plot(t, angular_momentum[2], color=colors[i])
    plt.ylabel("Angular momentum XYZ (kg⋅m2⋅s−1)")
    plt.xlabel("time")
    plt.legend()

plt.show()

for i, file in enumerate(files):
    file_path = open(f"{path_file}/{file}", "rb")
    data = pickle.load(file_path)
    print(file + "\n")
    print(data["status"])

    q = stack_states(data["states"], "q")
    qdot = stack_states(data["states"], "qdot")
    t = define_time(data["parameters"]["time"], data["n_shooting"])

    m = biorbd.Model(model)

    angular_momentum = angular_momentum_time_series(m, q, qdot)

    plt.plot(t, angular_momentum[1], label=data["n_shooting"], color=colors[i])
    plt.ylabel("Angular momentum Y (kg⋅m2⋅s−1)")
    plt.xlabel("time")
    plt.legend()

plt.show()


for i, file in enumerate(files):
    file_path = open(f"{path_file}/{file}", "rb")
    data = pickle.load(file_path)
    print(file + "\n")
    print(data["status"])

    q = stack_states(data["states"], "q")
    qdot = stack_states(data["states"], "qdot")
    qddot = stack_controls(data["controls"], "qddot")
    t = define_time(data["parameters"]["time"], data["n_shooting"])

    m = biorbd.Model(model)

    linear_momentum = linear_momentum_time_series(m, q, qdot)
    comdot = comdot_time_series(m, q, qdot)
    comddot = comddot_time_series(m, q, qdot, qddot)
    linear_momentum_rmsd = linear_momentum_deviation(m.mass(), comdot, t, comddot)

    plt.figure(1)
    plt.plot(t, comdot[2], label=data["n_shooting"], color=colors[i])

    plt.figure(2)
    plt.plot(t, linear_momentum_rmsd, label=data["n_shooting"], color=colors[i])
    plt.ylabel("Linear momentum RMSD (kg⋅m2⋅s−1)")
    plt.xlabel("time")
    plt.legend()

plt.show()
#
#
# path_file = "../../OnDynamicsForSommersaults_results/raw_explicit"
# model = "../Model_JeCh_15DoFs.bioMod"
# # ouvrir les fichiers
# files = os.listdir(path_file)
# files.sort()
# colors = pl.cm.jet(np.linspace(0, 1, len(files)))

# for i, file in enumerate(files):
#     file_path = open(f"{path_file}/{file}", "rb")
#     data = pickle.load(file_path)
#     print(file + "\n")
#     print(data["status"])
#
#     q = stack_states(data["states"], "q")
#     qdot = stack_states(data["states"], "qdot")
#     # t = define_time(data["parameters"]["time"], data["n_shooting"])
#
#     m = biorbd.Model(model)
#
#     angular_momentum = angular_momentum_time_series(m, q, qdot)
#
#     plt.plot(angular_momentum[2], label=i, color=colors[i])
#     plt.ylabel("Angular momentum Z (kg⋅m2⋅s−1)")
#     plt.xlabel("time")
#     plt.legend()
#
# plt.show()