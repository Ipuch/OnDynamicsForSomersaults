"""
This file is a demo to run optimal control problem of a miller with the chosen dynamics and with the 15-dof human model.
"""

import numpy as np
from bioptim import OdeSolver, CostType, InitialGuessList, Solver

from miller_ocp import MillerOcp
from miller_viz import add_custom_plots
from custom_dynamics.enums import MillerDynamics


def main(
    dynamics_type: MillerDynamics,
    thread: int = 8,
    solver: OdeSolver = OdeSolver.RK4,
    extra_obj: bool = False,
    n_shooting: tuple = (125, 25),
    initial_u: InitialGuessList = None,
    initial_x: InitialGuessList = None,
    phase_durations: tuple = None,
):
    """
    Main function for running the Miller optimal control problem with 15-dof human.

    Parameters
    ----------
    dynamics_type : MillerDynamics
        Type of dynamics to use.
    thread : int
        Number of threads to use.
    solver : OdeSolver
        Type of solver to use.
    extra_obj : bool
        Whether to use the extra objective (only for implicit dynamics).
    n_shooting : tuple
        Number of shooting nodes.
    initial_u : InitialGuessList
        Initial guess for the control.
    initial_x : InitialGuessList
        Initial guess for the states.
    phase_durations : tuple
        Time of each phase.
    """

    model_path = "Model_JeCh_15DoFs.bioMod"

    # --- Solve the program --- #
    miller = MillerOcp(
        biorbd_model_path=model_path,
        n_shooting=n_shooting,
        ode_solver=solver,
        dynamics_type=dynamics_type,
        n_threads=thread,
        somersaults=4 * np.pi,
        twists=6 * np.pi,
        use_sx=False,
        extra_obj=extra_obj,
        initial_u=initial_u,
        initial_x=initial_x,
        phase_durations=phase_durations,
    )

    add_custom_plots(miller.ocp, dynamics_type)
    miller.ocp.add_plot_penalty(CostType.CONSTRAINTS)
    np.random.seed(203)

    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1000)
    solver.set_print_level(5)
    solver.set_linear_solver("ma57")

    sol = miller.ocp.solve(solver)

    # --- Show results --- #
    sol.print_cost()
    # sol.graphs(show_bounds=True)
    # sol.animate(show_meshes=True)

    return miller.ocp, sol


if __name__ == "__main__":

    main(MillerDynamics.EXPLICIT, 4, OdeSolver.RK4(n_integration_steps=5), False)
    main(MillerDynamics.ROOT_EXPLICIT, 4, OdeSolver.RK4(n_integration_steps=5), False)
    main(MillerDynamics.IMPLICIT, 1, OdeSolver.RK2(n_integration_steps=5), True)
    main(MillerDynamics.ROOT_IMPLICIT, 1, OdeSolver.RK2(n_integration_steps=5), True)
    main(MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, 1, OdeSolver.RK4(n_integration_steps=5), True)
    main(MillerDynamics.ROOT_IMPLICIT_QDDDOT, 1, OdeSolver.RK4(n_integration_steps=5), True)
