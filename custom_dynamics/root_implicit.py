from typing import Union
import casadi as cas
from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    Node,
    PenaltyNodeList,
)
from bioptim.misc.enums import ConstraintType
from bioptim.interfaces.biorbd_interface import BiorbdInterface
from bioptim.limits.constraints import ImplicitConstraintFcn


def root_implicit_dynamic(
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
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    qddot = DynamicsFunctions.get(nlp.controls["qddot"], controls)

    return qdot, qddot


def custom_configure_root_implicit(ocp: OptimalControlProgram, nlp: NonLinearProgram):
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

    # we call the implicit constraint function in Bioptim but we could have called the custom one written below
    # ocp.implicit_constraints.add(
    #     ImplicitConstraintFcn.QDDOT_ROOT_EQUALS_ROOT_DYNAMICS,
    #     node=Node.ALL_SHOOTING,
    #     constraint_type=ConstraintType.IMPLICIT,
    #     phase=nlp.phase_idx,
    # )
    ocp.implicit_constraints.add(
        implicit_root_dynamics,
        node=Node.ALL_SHOOTING,
        constraint_type=ConstraintType.IMPLICIT,
        phase=nlp.phase_idx,
    )

    ConfigureProblem.configure_dynamics_function(ocp, nlp, root_implicit_dynamic)


def implicit_root_dynamics(all_pn: PenaltyNodeList, **unused_param):
    """
    Compute the difference between symbolic joint torques and inverse dynamic results
    It does not include any inversion of mass matrix

    Parameters
    ----------
    all_pn: PenaltyNodeList
        The penalty node elements
    **unused_param: dict
        Since the function does nothing, we can safely ignore any argument
    """

    nlp = all_pn.nlp
    nb_root = nlp.model.nbRoot()

    q = nlp.states["q"].mx
    qdot = nlp.states["qdot"].mx
    qddot = nlp.states["qddot"].mx if "qddot" in nlp.states.keys() else nlp.controls["qddot"].mx

    if nlp.external_forces:
        raise NotImplementedError(
            "This implicit constraint implicit_root_dynamics is not implemented yet with external forces"
        )

    floating_base_constraint = nlp.model.InverseDynamics(q, qdot, qddot).to_mx()[:nb_root]

    var = []
    var.extend([nlp.states[key] for key in nlp.states])
    var.extend([nlp.controls[key] for key in nlp.controls])
    var.extend([param for param in nlp.parameters])

    return BiorbdInterface.mx_to_cx("FloatingBaseConstraint", floating_base_constraint, *var)
