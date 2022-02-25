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

path_file = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw_we_11-02-11"
# path_file = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw_08-02-22"
# path_file = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw_05-02-22"
model = "../Model_JeCh_15DoFs.bioMod"
# ouvrir les fichiers
files = os.listdir(path_file)
files.sort()

colors = pl.cm.jet(np.linspace(0, 1, len(files)))
t_all_ex = np.zeros((0))
t_all_rex = np.zeros((0))
for i, file in enumerate(files):
    if ".pckl" in file:
        file_path = open(f"{path_file}/{file}", "rb")
        data = pickle.load(file_path)
        t = data["computation_time"]
        if data["status"] == 0 and "explicit" == data["dynamics_type"]:
            t_all_ex = np.hstack((t_all_ex, t))
        if data["status"] == 0 and "root_explicit" == data["dynamics_type"]:
            t_all_rex = np.hstack((t_all_rex, t))

plt.plot(t_all_ex / 3600, label="explicit", marker=".")
plt.plot(t_all_rex / 3600, label="root_explicit", marker=".")
plt.legend()

plt.show()
print("hey")
