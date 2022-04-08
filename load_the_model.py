"""
This file is to display the human model into bioviz
"""
import bioviz


export_model = True
background_color = (1, 1, 1) if export_model else (0.5, 0.5, 0.5)
show_gravity_vector = False if export_model else True
show_floor = False if export_model else True
show_local_ref_frame = False if export_model else True
show_global_ref_frame = False if export_model else True
show_markers = False if export_model else True
show_mass_center = False if export_model else True
show_global_center_of_mass = False if export_model else True
show_segments_center_of_mass = False if export_model else True

model_name = "Model_JeCh_15DoFs.bioMod"
biorbd_viz = bioviz.Viz(
    model_name,
    show_gravity_vector=show_gravity_vector,
    show_floor=show_floor,
    show_local_ref_frame=show_local_ref_frame,
    show_global_ref_frame=show_global_ref_frame,
    show_markers=show_markers,
    show_mass_center=show_mass_center,
    show_global_center_of_mass=show_global_center_of_mass,
    show_segments_center_of_mass=show_segments_center_of_mass,
    mesh_opacity=1,
    background_color=background_color,
)
biorbd_viz.set_camera_position(-0.5, 3.5922578781963685, 0.1)
biorbd_viz.resize(1000, 1000)
if export_model:
    biorbd_viz.snapshot("doc/model.png")
biorbd_viz.exec()
