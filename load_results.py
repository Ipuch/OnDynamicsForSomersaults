"""
This files loads the results of the simulation, plots them, and animates them.
"""
import numpy as np
import pickle
from bioptim import OptimalControlProgram

# filename = "raw/miller_MillerDynamics.EXPLICIT_irand10_extraobjFalse_125_25.pckl"  # or .bo, any file in raw.
filename = "raw/miller_MillerDynamics.ROOT_IMPLICIT_irand0_extraobjTrue_125_25.bo"


def show_animation(model_path: str, q: np.ndarray, fixed_floating_base: bool = False):
    """
    Show the animation of the model

    Parameters
    ----------
    model_path : str
        Model to animate
    q : np.ndarray
        Array of the states to animate
    fixed_floating_base : bool
        If the floating base is fixed or not when the motion is played

    Returns
    -------
    None
    """
    import bioviz

    manually_animate = False
    biorbd_viz = bioviz.Viz(model_path, show_floor=False, show_gravity_vector=False)

    if fixed_floating_base:
        q[:6, :] = 0

    if manually_animate:
        i = 0
        while biorbd_viz.vtk_window.is_active:
            biorbd_viz.set_q(q[:, i])
            i = i + 1
    else:
        biorbd_viz.load_movement(q)
        biorbd_viz.exec()


if filename.endswith(".bo"):
    try:
        ocp, sol = OptimalControlProgram.load(filename)
        sol.print_cost()
        sol.animate(n_frames=100)
    except:
        model_name = "Model_JeCh_15DoFs.bioMod"
        file = open(f"{filename}", "rb")
        data = pickle.load(file)
        q = np.hstack((data[0][0]["q"], data[0][1]["q"]))
        show_animation(model_name, q)

elif filename.endswith(".pckl"):
    model_name = "Model_JeCh_15DoFs.bioMod"
    file = open(f"{filename}", "rb")
    data = pickle.load(file)
    q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))

    # Animate the model
    show_animation(model_name, q)
else:
    raise ValueError("filename must end with .bo or .pkl")
