"""
This script is used to generate the videos of Miller optimal control problems.
It requires the dataframe of all results to run the script and dataframe.
"""

from bioptim import OptimalControlProgram
from custom_dynamics.enums import MillerDynamics
import os
import pandas as pd
from videos import generate_video

# all data path
my_path = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw_simulation_data/"
# save data path
save_path = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/figures/V5/somersaults/"

# Load dataframe and select main results in the main cluster
df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")
df_results = df_results[df_results["status"] == 0]
# only the one that were in the main cluster
df_results = df_results[df_results["main_cluster"] == True]
# only one trial by cluster
for d in MillerDynamics:
    df_results = df_results.drop(df_results[df_results["dynamics_type"] == d].index[1:])
grps = ["Explicit", "Root_Explicit", "Implicit_qddot", "Root_Implicit_qddot", "Implicit_qdddot", "Root_Implicit_qdddot"]
df_results["grps"] = pd.Categorical(df_results["grps"], grps)
df_results = df_results.sort_values("grps")

# open files in the data path
all_files = os.listdir(my_path)
all_files.sort()
files = []
for index, row in df_results.iterrows():
    d = row.dynamics_type
    rand_n = row.irand
    for i, file in enumerate(all_files):

        if f".{d.name}_i" in file:
            if file.endswith(".bo") and f"irand{rand_n}" in file:
                files.append(file)

for f in files[3:]:
    print(f)
    ocp, solution = OptimalControlProgram.load(my_path + f)
    generate_video(solution, save_path, f.replace(".bo", "").replace(".", "_"))
