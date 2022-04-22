"""
This script is used to generate the kinograms of Miller optimal control problems.
It requires the dataframe of all results to run the script and dataframe.
"""

import bioviz
from bioptim import OptimalControlProgram
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from custom_dynamics.enums import MillerDynamics
import os
import pandas as pd

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
        print(i)
        if f".{d.name}_i" in file:
            if file.endswith(".bo") and f"irand{rand_n}" in file:
                files.append(file)

# number of frames considered with steps
start = 0
step = 15
end = 151

for i, f in enumerate(files):
    ocp, solution = OptimalControlProgram.load(my_path + f)
    s = f.split(".")
    s = s[1].split("_irand")
    s = s[0]
    b: bioviz.Viz = solution.animate(
        show_now=False,
        show_meshes=True,
        show_global_center_of_mass=False,
        show_gravity_vector=False,
        show_floor=False,
        show_segments_center_of_mass=False,
        show_global_ref_frame=False,
        show_local_ref_frame=False,
        show_markers=False,
        show_muscles=False,
        show_wrappings=False,
        background_color=(1, 1, 1),
        mesh_opacity=0.97,
    )[0]

    b.resize(600, 1050)

    # Position camera
    b.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
    b.set_camera_roll(90)
    b.set_camera_zoom(0.308185240948253)
    b.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)

    print("roll")
    print(b.get_camera_roll())
    print("zoom")
    print(b.get_camera_zoom())
    print("position")
    print(b.get_camera_position())
    print("get_camera_focus_point")
    print(b.get_camera_focus_point())

    fig, ax = plt.subplots(1, 11, figsize=(19.20, 5.4))
    fig.subplots_adjust(hspace=0, wspace=0)

    # Taking snapshot
    count = 0
    for snap in range(start, end, step):
        b.movement_slider[0].setValue(snap)
        b.snapshot(f"{save_path}/{s}_{snap}.png")
        # b.refresh_window()
        img = mpimg.imread(f"{save_path}/{s}_{snap}.png")

        ax[count].xaxis.set_major_locator(plt.NullLocator())
        ax[count].yaxis.set_major_locator(plt.NullLocator())
        ax[count].imshow(img)
        ax[count].spines["top"].set_visible(False)
        ax[count].spines["right"].set_visible(False)
        ax[count].spines["bottom"].set_visible(False)
        ax[count].spines["left"].set_visible(False)
        count += 1

    b.quit()
    fig.show()
    fig.savefig(f"{save_path}/kinogram_{s}.svg", format="svg", dpi=900, bbox_inches="tight")
    fig.savefig(f"{save_path}/kinogram_{s}.png", format="png", dpi=900, bbox_inches="tight")
    plt.close(fig)
