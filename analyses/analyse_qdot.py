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

path_file = "../../OnDynamicsForSommersaults_results/raw_04-02-22"
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
        qdot = stack_states(data["states"], "qdot")
        qdot_diff = np.diff(qdot)
        t = define_time(data["parameters"]["time"], data["n_shooting"])

        plt.plot(t[:-1], qdot_diff[7, :], color=colors[i])

plt.ylabel("Residual torques norm xyz Rx Ry Rz (N or Nm)")
plt.xlabel("time")
plt.legend()
# plt.yscale("log")
plt.show()
