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

path_file = "../../OnDynamicsForSommersaults_results/raw"
model = "../Model_JeCh_15DoFs.bioMod"
# ouvrir les fichiers
files = os.listdir(path_file)
files.sort()

colors = pl.cm.jet(np.linspace(0, 1, len(files)))
qdot_init = np.zeros((6, 1))
for i, file in enumerate(files):
    file_path = open(f"{path_file}/{file}", "rb")
    data = pickle.load(file_path)
    print(file + "\n")
    print(data["status"])
    if data["status"] == 0 and "explicit" in data["dynamics_type"]:
        qdot_init = np.hstack((qdot_init, data["states"][0]["qdot"][0:1, :6].T))

plt.plot(qdot_init[0, 1:], label="x_velocity", marker=".")
plt.plot(qdot_init[1, 1:], label="y_velocity", marker=".")
plt.plot(qdot_init[2, 1:], label="z_velocity", marker=".")
plt.plot(qdot_init[3, 1:], label="salto", marker="o")
plt.plot(qdot_init[4, 1:], label="tilt", marker="o")
plt.plot(qdot_init[5, 1:], label="twist", marker="o")
plt.legend()

plt.show()
print("hey")
