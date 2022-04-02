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
)
import biorbd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np

path_file = "../../OnDynamicsForSommersaults_results/raw_test_threads"
# folders = ["explicit", "root_implicit", "implicit"]
folders = ["root_explicit"]
model = "../Model_JeCh_15DoFs.bioMod"
# ouvrir les fichiers
files = os.listdir(path_file)
files.sort()

colors = pl.cm.jet(np.linspace(0, 1, len(folders)))
for j, folder in enumerate(folders):
    full_path = path_file + "/" + folder
    files = os.listdir(full_path)
    files.sort()
    for i, file in enumerate(files):
        file_path = open(f"{full_path}/{file}", "rb")
        data = pickle.load(file_path)
        print(file + "\n")
        print(data["status"])

        n_thread = data["n_theads"]
        t = data["computation_time"] / (32 / n_thread)

        label = None
        if i == 0:
            label = folder

        plt.plot(n_thread, t, label=label, color=colors[j], marker="o")

plt.ylabel("time")
plt.xlabel("Threads")
# plt.yscale('log')
plt.legend()
plt.show()

print("he")
