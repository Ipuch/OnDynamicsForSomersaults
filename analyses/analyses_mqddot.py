import os
import pickle
from utils import (
    stack_states,
    stack_controls,
    define_time,
    angular_momentum_deviation,
    angular_momentum_time_series,
    linear_momentum_time_series,
    linear_momentum_deviation,
    comdot_time_series,
    comddot_time_series,
    residual_torque_time_series,
)
import biorbd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np

path_file = "../../OnDynamicsForSommersaults_results/raw_08-02-22/miller_explicit_i_rand0.pckl"
model = "../Model_JeCh_15DoFs.bioMod"
m = biorbd.Model(model)
file_path = open(f"{path_file}", "rb")
data = pickle.load(file_path)
q = stack_states(data["states"], "q")
qdot = stack_states(data["states"], "qdot")
tau = stack_controls(data["controls"], "tau")

qddot = np.zeros((15, 151))
for i in range(q.shape[1]):
    qddot[:, i] = m.ForwardDynamics(q[:,i], qdot[:,i], tau[:,i]).to_array()

# test = np.zeros((6, 151))
# for i in range(q.shape[1]):
#     test[:, i] = m.massMatrix(q[:, i], True).to_array()[:6, :] @ qdot[:, i]
#
# plt.plot(test.T)
# plt.show()

test = np.zeros((6, 151))
for i in range(q.shape[1]):
    test[:, i] = m.massMatrix(q[:, i], True).to_array()[:6, :] @ qddot[:, i] \
                 + m.NonLinearEffect(q[:, i], qdot[:, i]).to_array()[:6]

plt.plot(test[0, :])
plt.show()

# m = biorbd.Model(model)
#
# residual_torque_norm = np.linalg.norm(residual_torque_time_series(m, q, qdot, qddot), axis=0)
# for ii in range(1):
#     if ii == 0:
#         plt.plot(t, residual_torque_norm, label=data["n_shooting"], color=colors[i])
#     else:
#         plt.plot(t, residual_torque_norm[ii], color=colors[i])
#
# plt.ylabel("Residual torques norm xyz Rx Ry Rz (N or Nm)")
# plt.xlabel("time")
# plt.legend()
# plt.yscale("log")
# plt.show()
#
# # is it non sensitive to
# v = np.hstack((np.random.random(6),np.ones(9)))
# w = np.hstack((np.random.random(6),np.ones(9)))
# (m.massMatrix(v).to_array() - m.massMatrix(w).to_array())[:6,:6] == 0


