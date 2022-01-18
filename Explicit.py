"""
This is an example of how to state the explicit formulation of a problem.
The avatar must complete two somersault and three twist rotations while minimizing the hand trajectories and the joint
velocity variations.
"""

import numpy as np
import biorbd_casadi as biorbd
import casadi as cas
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Bounds,
    ConstraintFcn,
    ObjectiveFcn,
    BiMappingList,
    ConstraintList,
    InitialGuessList,
    InterpolationType,
    ObjectiveList,
    Node,
    DynamicsList,
    BoundsList,
    OdeSolver,
    PenaltyNode,
    BiorbdInterface,
    Solver,
    CostType,
)


def prepare_ocp_explicit(biorbd_model_path: str, n_shooting: int, ode_solver: Solver) -> OptimalControlProgram:
    """
    Prepare the explicit version of the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    n_shooting: int
        The number of shooting points
    ode_solver: Solver
        The solver to be used while solving the OCP
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -200, 200, 0

    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque() - biorbd_model.nbRoot()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key="qdot", weight=5)  # Regularization
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, reference_jcs=3, marker_index=6, weight=1000)  # Right hand trajetory
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, reference_jcs=7, marker_index=11, weight=1000)  # Left hand trajectory

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN) # expand=False

    # Initial guesses
    duration = 1.545  # Real data : 103 * 1/200 * 3,
    vertical_velocity_0 = 9.2 # Real data
    somerault_rate_0 = 4 * np.pi / duration

    x = np.vstack((np.zeros((nb_q, n_shooting + 1)), np.ones((nb_qdot, n_shooting + 1))))

    x[2, :] = (
        vertical_velocity_0 * np.linspace(0, duration, n_shooting + 1) + -9.81 / 2 * np.linspace(0, duration, n_shooting + 1) ** 2
    )
    x[3, :] = np.linspace(0, 4 * np.pi, n_shooting + 1)
    x[5, :] = np.linspace(0, 4 * np.pi, n_shooting + 1)
    # x[7, :] = np.random.random((1, n_shooting + 1)) * np.pi/2 - (np.pi - np.pi/4)
    # x[9, :] = np.random.random((1, n_shooting + 1)) * np.pi/2 + np.pi/4

    x[7, :] = - np.pi/2 + np.pi/4 * np.sin(np.linspace(0, (2*np.pi), n_shooting+1) * duration * 2) \
              + (np.random.random((1, n_shooting + 1))-0.5) * np.pi / 10

    x[9, :] = np.pi/2 + np.pi/4 * np.sin(np.linspace(0, (2*np.pi), n_shooting+1) * duration * 2) \
              - (np.random.random((1, n_shooting + 1))-0.5) * np.pi / 10

    x[nb_q + 2, :] = vertical_velocity_0 - 9.81 * np.linspace(0, duration, n_shooting + 1)
    x[nb_q + 3, :] = somerault_rate_0
    x[nb_q + 5, :] = 4 * np.pi / duration

    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.EACH_FRAME)

    # Path constraint
    # revoir les bornes que Mickael veut ########################################################################### !!!
    x_bounds = BoundsList()
    x_min = np.zeros((nb_q + nb_qdot, 3))
    x_max = np.zeros((nb_q + nb_qdot, 3))
    x_min[:, 0] = [0, 0, 0, 0, 0, 0, 0, -2.8, 0, 2.8,
                   -1, -1, vertical_velocity_0 - 2, somerault_rate_0 - 1, 0, 0, 0, 0, 0, 0]
    x_max[:, 0] = [0, 0, 0, 0, 0, 0, 0, -2.8, 0, 2.8,
                   1, 1, vertical_velocity_0 + 2, somerault_rate_0 + 1, 0, 0, 0, 0, 0, 0]
    x_min[:, 1] = [-3, -3, -0.001, -0.001, -np.pi / 4, -np.pi, -1, -np.pi + 0.01, -1.27, 0.01,
                   -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 1] = [3, 3, 10, 4 * np.pi + 0.1, np.pi / 4, 6 * np.pi + 0.1, 1.27, -0.01, 1, np.pi - 0.01,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    x_min[:, 2] = [ -0.1, -0.1, -0.1, 4 * np.pi - 0.1, -15 * np.pi / 180, 4 * np.pi - 0.1, -1, -np.pi + 0.01, -1.27, 0.01,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 2] = [0.1, 0.1, 0.1, 4 * np.pi + 0.1, 15 * np.pi / 180, 6 * np.pi + 0.1, 1.27, -0.01, 1, np.pi - 0.01,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    x_bounds.add(bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * nb_tau, [tau_max] * nb_tau)

    mapping = BiMappingList()
    mapping.add("tau", [None, None, None, None, None, None, 0, 1, 2, 3], [6, 7, 8, 9])

    u_init = InitialGuessList()
    u_init.add([tau_init] * nb_tau)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=duration-0.5, max_bound=duration+0.5)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        duration,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        n_threads=8,
        variable_mappings=mapping,
        ode_solver=ode_solver,
    )



np.random.seed(0)

solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
solver.set_maximum_iterations(10000)
ocp = prepare_ocp_explicit("Model_JeCh_10DoFs.bioMod", n_shooting=150, ode_solver=OdeSolver.RK4())
# solver.set_convergence_tolerance(1e-2)
# solver.set_acceptable_constr_viol_tol(1e-2)

ocp.add_plot_penalty(CostType.ALL)
ocp.print(to_console=False, to_graph=False)

sol = ocp.solve(solver)
sol.animate(nb_frames=-1, show_meshes=False) # show_mesh=True
