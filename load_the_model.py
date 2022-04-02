"""
This file is to display the human model into bioviz
"""
import bioviz

model_name = "Model_JeCh_15DoFs.bioMod"
biorbd_viz = bioviz.Viz(model_name)
biorbd_viz.exec()
