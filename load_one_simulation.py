
import bioviz
import numpy as np
from bioptim import OptimalControlProgram

model_name = "Model_JeCh_10DoFs.bioMod"
biorbd_viz = bioviz.Viz(model_name)
file = "Model_JeCh_10DoFs.bo"
# Loading
ocp, sol = OptimalControlProgram.load(file)

q = sol.states["q"]

# Animate the mode
manually_animate = False
if manually_animate:
    i = 0
    while biorbd_viz.vtk_window.is_active:
        biorbd_viz.set_q(q[:, i])
        i = (i+1)
else:
    biorbd_viz.load_movement(q)
    biorbd_viz.exec()
