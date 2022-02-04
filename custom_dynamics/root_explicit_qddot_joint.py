from typing import Union
import casadi as cas
from casadi import MX, solve, Function
from bioptim import (
    OptimalControlProgram,
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
    nb_root = nlp.model.nbRoot()

    qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joint"], controls)

    mass_matrix_nl_effects = nlp.model.InverseDynamics(
        q, qdot, cas.vertcat(cas.MX.zeros((nb_root, 1)), qddot_joints)
    ).to_mx()[:6]
    mass_matrix_inverse = nlp.model.massMatrixInverse(q).to_mx()
    mass_matrix = nlp.model.massMatrix(q).to_mx()
    # mass_matrix_inverse = solve(mass_matrix[: nb_root, : nb_root], MX.eye(nb_root), "qr")

    mass_matrix = nlp.model.massMatrix(q).to_mx()
    # M_func = cas.Function("M_func", [q], [mass_matrix], ["q"], ["M"]).expand()
    # N_func = cas.Function("N_func", [q, qdot], [mass_matrix], ["q"], ["M"]).expand()

    # qddot_root = -M_BB^-1 * ( M_BJ  * qddot_joints + N_B )
    # qddot_root = -mass_matrix_inverse[:nb_root, :nb_root] @ mass_matrix_nl_effects[:nb_root]
    # qddot_root = ldl_solve(mass_matrix_inverse[:nb_root, :nb_root], mass_matrix_nl_effects[:nb_root])

    # qddot_root = solve(mass_matrix[: nb_root, : nb_root], MX.eye(nb_root), "ldl") @ mass_matrix_nl_effects[: nb_root]
    # qddot_root = solve(mass_matrix[: nb_root, : nb_root], mass_matrix_nl_effects[: nb_root], "ldl")
    # q_sym = MX.sym("q_sym", nlp.model.nbQ(), 1)
    # qdot_sym = MX.sym("q_sym", nlp.model.nbQdot(), 1)
    # qddot_sym = MX.sym("q_sym", nlp.model.nbQddot(), 1)

    mass_matrix_nl_effects_func = Function("mass_matrix_nl_effects_func", [q, qdot, qddot_joints],
                                           [mass_matrix_nl_effects[:nb_root]]
                                           ).expand()

    M_66 = mass_matrix[: nb_root, : nb_root]
    M_66_func = Function("M66_func", [q], [M_66]).expand()

    qddot_root = solve(- M_66_func(q), mass_matrix_nl_effects_func(q, qdot, qddot_joints), "ldl")

    # qddot_root = solve(mass_matrix[: nb_root, : nb_root], MX.eye(nb_root), "ldl") @ mass_matrix_nl_effects_func(q, qdot, qddot)

    return qdot, cas.vertcat(qddot_root, qddot_joints)


def custom_configure_root_explicit(ocp: OptimalControlProgram, nlp: NonLinearProgram):
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
    configure_qddot_joint(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, root_explicit_dynamic)


def configure_qddot_joint(nlp, as_states: bool, as_controls: bool):
    """
    Configure the generalized accelerations

    Parameters
    ----------
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized velocities should be a state
    as_controls: bool
        If the generalized velocities should be a control
    """
    nb_root = nlp.model.nbRoot()
    name_qddot_joint = [str(i + nb_root) for i in range(nlp.model.nbQddot() - nb_root)]
    ConfigureProblem.configure_new_variable("qddot_joint", name_qddot_joint, nlp, as_states, as_controls)