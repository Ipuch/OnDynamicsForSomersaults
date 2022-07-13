import numpy as np
import biorbd
from bioptim import OdeSolver, InitialGuessList, InterpolationType

from custom_dynamics.enums import MillerDynamics
from analyses.utils import root_explicit_dynamics
from miller import main


def to_initial_guess(v: dict):
    """
    Convert a dictionary of values to a list of initial guess.

    Parameters
    ----------
    v : dict
        Dictionary of values.

    Returns
    -------
    InitialGuessList
        List of initial guess.
    """
    initial_guess = InitialGuessList()

    for stage in v:
        initial_guess.add(stage["all"], interpolation=InterpolationType.EACH_FRAME)

    return initial_guess


if __name__ == "__main__":
    ocp_exp, sol_exp = main(MillerDynamics.EXPLICIT, 4, OdeSolver.RK4(n_integration_steps=5), False)
    ocp_root_exp, sol_root_exp = main(MillerDynamics.ROOT_EXPLICIT, 4, OdeSolver.RK4(n_integration_steps=5), False)

    # build an initial guess from the solution of the root explicit OCP
    initial_x = to_initial_guess(sol_root_exp.states)

    # compute inverse dynamics from the solution of the root explicit OCP
    model = biorbd.Model(sol_root_exp.ocp.nlp[0].model.path().absolutePath().to_string())
    tau = [np.zeros((u["all"].shape[0], u["all"].shape[1])) for u in sol_root_exp.controls]
    for i, tau_i in enumerate(tau):
        for j, tau_ij in enumerate(tau_i):
            q = sol_root_exp.states[i]["q"][:, j]
            qdot = sol_root_exp.states[i]["qdot"][:, j]
            qddot_joint = sol_root_exp.controls[i]["qddot_joint"][:, j]
            qddot_base = root_explicit_dynamics(m=model, q=q, qdot=qdot, qddot_joints=qddot_joint)
            qddot = np.concatenate((qddot_base, qddot_joint))
            tau[i][:, j] = model.InverseDynamics(q, qdot, qddot).to_array()[6::]
    # fill the InitialGuessList with the tau computed from the inverse dynamics
    initial_u = InitialGuessList()
    for tau_i in tau:
        initial_u.add(tau_i[:, :-1], interpolation=InterpolationType.EACH_FRAME)

    ocp_exp2, sol_exp_2 = main(
        MillerDynamics.EXPLICIT,
        4,
        OdeSolver.RK4(n_integration_steps=5),
        False,
        initial_u=initial_u,
        initial_x=initial_x,
        phase_durations=tuple(sol_root_exp.parameters["time"].toarray().T[0].tolist()),
    )

    ocp_exp.save(sol_exp, f"miller_exp.bo", stand_alone=True)
    ocp_root_exp.save(sol_root_exp, f"miller_root_exp.bo", stand_alone=True)
    ocp_exp2.save(sol_exp_2, f"miller_exp_2.bo", stand_alone=True)
