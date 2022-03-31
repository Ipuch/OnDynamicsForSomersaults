import bioviz
import numpy as np
from bioptim import OptimalControlProgram
import pickle

file_type = "bo"

if file_type == "bo":
    model_name = "Model_JeCh_15DoFs.bioMod"
    file = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/other/test_min_qddot.bo"
    # Loading
    ocp, sol = OptimalControlProgram.load(file)
    data = sol.states
    q = np.hstack((data[0]["q"], data[1]["q"]))
elif file_type == "pckl":
    # file_name = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/miller_explicit_irand/miller_explicit_irand1.pckl"
    file_name = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw_with_min_qddot/test_min_qddot.pckl"
    model_name = "Model_JeCh_15DoFs.bioMod"
    file = open(f"{file_name}", "rb")
    data = pickle.load(file)
    q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))

# q[:6, :] = 0
# Animate the model
manually_animate = False
biorbd_viz = bioviz.Viz(model_name, show_floor=False, show_gravity_vector=False)
if manually_animate:
    i = 0
    while biorbd_viz.vtk_window.is_active:
        biorbd_viz.set_q(q[:, i])
        i = i + 1
else:
    biorbd_viz.load_movement(q)
    biorbd_viz.exec()
