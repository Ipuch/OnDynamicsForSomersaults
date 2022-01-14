"""
This is an example of how to state the explicit formulation of a problem.
The avatar must complete two somersault and three twist rotations while minimizing the hand trajectories and the joint
velocity variations.
"""

import numpy as np
from typing import Union
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
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
)


def root_explicit_dynamic(
    states: Union[cas.MX, cas.SX],
    controls: Union[cas.MX, cas.SX],
    parameters: Union[cas.MX, cas.SX],
    nlp: NonLinearProgram,
) -> tuple:
    """
    Parameters
    ----------
    states: Union[MX, SX]
        The state of the system
    controls: Union[MX, SX]
        The controls of the system
    parameters: Union[MX, SX]
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[Union[MX, SX]] format
    """

    DynamicsFunctions.apply_parameters(parameters, nlp)
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    qddot_joints = DynamicsFunctions.get(nlp.controls["qddot"], controls)[nlp.model.nbRoot():] # comment ca se fait que ca c'est [0, 0, 0, 0, 0, 0, VARS] ?

    mass_matrix = nlp.model.massMatrix(q).to_mx()
    nl_effects = nlp.model.NonLinearEffect(q, qdot).to_mx()
    # make sure of the index of mass_matrix[:nlp.model.nbRoot(), nlp.model.nbRoot():]
    # qddot_root = cas.inv(mass_matrix[:nlp.model.nbRoot(), :nlp.model.nbRoot()]) @ \
    #              (-mass_matrix[:nlp.model.nbRoot(), nlp.model.nbRoot():] @ qddot_joints - nl_effects[:nlp.model.nbRoot()])
    # qddot_root = cas.solve(mass_matrix[:nlp.model.nbRoot(), :nlp.model.nbRoot()], cas.MX.eye(nlp.model.nbRoot())) @ \
    #              (-mass_matrix[:nlp.model.nbRoot(), nlp.model.nbRoot():] @ qddot_joints - nl_effects[:nlp.model.nbRoot()])
    qddot_root = cas.solve(mass_matrix[:nlp.model.nbRoot(), :nlp.model.nbRoot()], cas.MX.eye(nlp.model.nbRoot()), 'ldl') @ \
                 (-mass_matrix[:nlp.model.nbRoot(), nlp.model.nbRoot():] @ qddot_joints - nl_effects[:nlp.model.nbRoot()])

    return qdot, cas.vertcat(qddot_root, qddot_joints)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qddot(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, root_explicit_dynamic)


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
    qddot_min, qddot_max, qddot_init = -200, 200, 0

    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_qddot = nb_qdot - biorbd_model.nbRoot()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key="qdot", weight=1)  # Regularization
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, reference_jcs=3, marker_index=6, weight=1000)  # Right hand trajetory
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, reference_jcs=7, marker_index=11, weight=1000)  # Left hand trajectory

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=root_explicit_dynamic, expand=False)

    # Initial guesses
    duration = 1.545  # Real data : 103 * 1/200 * 3
    vertical_velocity_0 = 9.2 # Real data
    somerault_rate_0 = 4 * np.pi / duration
    
    x = np.vstack((np.zeros((nb_q, n_shooting + 1)), np.ones((nb_qdot, n_shooting + 1))))
    x[2, :] = (
        vertical_velocity_0 * np.linspace(0, duration, n_shooting + 1) + -9.81 / 2 * np.linspace(0, duration, n_shooting + 1) ** 2
    )
    x[3, :] = np.linspace(0, 4 * np.pi, n_shooting + 1)
    x[5, :] = np.linspace(0, 4 * np.pi, n_shooting + 1)
    x[7, :] = np.random.random((1, n_shooting + 1)) * np.pi - np.pi
    x[9, :] = np.random.random((1, n_shooting + 1)) * np.pi

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
                   -1, -1, 7, x[nb_q + 3, 0], 0, x[nb_q + 5, 0], 0, 0, 0, 0]
    x_max[:, 0] = [0, 0, 0, 0, 0, 0, 0, -2.8, 0, 2.8,
                   1, 1, 10, x[nb_q + 3, 0], 0, x[nb_q + 5, 0], 0, 0, 0, 0]
    x_min[:, 1] = [-1, -1, -0.001, -0.001, -np.pi / 4, -np.pi, -1, -np.pi, -1.27, 0,
                   -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 1] = [1, 1, 10, 4 * np.pi + 0.1, np.pi / 4, 4 * np.pi + 0.1, 1.27, 0, 1, np.pi,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    x_min[:, 2] = [ -0.1, -0.1, -0.1, 4 * np.pi - 0.1, -15 * np.pi / 180, 4 * np.pi - 0.1, -1, -np.pi, -1.27, 0,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 2] = [0.1, 0.1, 0.1, 4 * np.pi + 0.1, 15 * np.pi / 180, 4 * np.pi + 0.1, 1.27, 0, 1, np.pi,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    x_bounds.add(bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([qddot_min] * nb_qddot, [qddot_max] * nb_qddot)

    mapping = BiMappingList()
    mapping.add("qddot", [None, None, None, None, None, None, 0, 1, 2, 3], [6, 7, 8, 9])

    u_init = InitialGuessList()
    u_init.add([qddot_init] * nb_qddot)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=duration-0.3, max_bound=duration+0.3)

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
        n_threads=4,
        variable_mappings=mapping,
        ode_solver=ode_solver,
    )



np.random.seed(0)

solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
ocp = prepare_ocp_explicit("Model_JeCh_10DoFs.bioMod", n_shooting=100, ode_solver=OdeSolver.RK4())
# solver.set_convergence_tolerance(1e-2)
# solver.set_acceptable_constr_viol_tol(1e-2)
# solver.set_maximum_iterations(1000)

ocp.add_plot_penalty(CostType.ALL)
ocp.print(to_console=False, to_graph=False)

sol = ocp.solve(solver)
sol.animate(nb_frames=-1, show_meshes=False) # show_mesh=True
