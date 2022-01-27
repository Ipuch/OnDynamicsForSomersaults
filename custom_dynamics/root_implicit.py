from typing import Union
import casadi as cas
from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    Node,
)
from bioptim.limits.constraints import ImplicitConstraintFcn
from bioptim.misc.enums import ConstraintType


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

    ocp.implicit_constraints.add(
        ImplicitConstraintFcn.QDDOT_ROOT_EQUALS_ROOT_DYNAMICS,
        node=Node.ALL_SHOOTING,
        constraint_type=ConstraintType.IMPLICIT,
        phase=nlp.phase_idx,
    )

    ConfigureProblem.configure_dynamics_function(ocp, nlp, root_implicit_dynamic)
