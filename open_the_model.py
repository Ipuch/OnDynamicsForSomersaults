import bioviz
import numpy as np

model_name = "Model_JeCh_10DoFs.bioMod"
biorbd_viz = bioviz.Viz(model_name)

# Create a movement
n_frames = 2
q = np.zeros((biorbd_viz.nQ, n_frames))
for ii in range(biorbd_viz.nQ):
    q[ii, :] = np.linspace(-1, 1, n_frames)

# Animate the mode
manually_animate = False
if manually_animate:
    i = 0
    while biorbd_viz.vtk_window.is_active:
        biorbd_viz.set_q(q[:, i])
        i = (i + 1) % n_frames
else:
    biorbd_viz.load_movement(q)
    biorbd_viz.exec()
