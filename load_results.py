"""
This files loads the results of the simulation, plots them, and animates them.
"""
import numpy as np
import pickle
from bioptim import OptimalControlProgram
import bioviz

filename = "raw/miller_MillerDynamics.EXPLICIT_irand10_extraobjFalse_125_25.pckl"  # or .bo, any file in raw.

if filename.endswith(".bo"):
    ocp, sol = OptimalControlProgram.load(filename)
    sol.graphs()
    sol.animate()

elif filename.endswith(".pckl"):
    model_name = "Model_JeCh_15DoFs.bioMod"
    file = open(f"{filename}", "rb")
    data = pickle.load(file)
    q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))

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
else:
    raise ValueError("filename must end with .bo or .pkl")
