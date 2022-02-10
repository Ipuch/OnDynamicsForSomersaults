from bioptim import PhaseTransition, NonLinearProgram
from casadi import MX
import biorbd_casadi as biorbd


def custom_phase_transition(
    transition: PhaseTransition,
    nlp_pre: NonLinearProgram,
    nlp_post: NonLinearProgram,
) -> MX:
    """
    The constraint of the transition which maintain the difference between angular momentum to zero.

    Parameters
    ----------
    transition: PhaseTransition
        The placeholder for the transition
    nlp_pre: NonLinearProgram
        The nonlinear program of the pre phase
    nlp_post: NonLinearProgram
        The nonlinear program of the post phase

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    q_pre = nlp_pre.states["q"].mx
    q_post = nlp_post.states["q"].mx

    qdot_pre = nlp_pre.states["qdot"].mx
    qdot_post = nlp_post.states["qdot"].mx

    pre_angular_momentum = nlp_pre.model.angularMomentum(q_pre, qdot_pre).to_mx()
    post_angular_momentum = nlp_post.model.angularMomentum(q_post, qdot_post).to_mx()

    pre_states_cx = nlp_pre.states.cx_end
    post_states_cx = nlp_post.states.cx

    func = biorbd.to_casadi_func(
        "angular_momentum_transition",
        pre_angular_momentum - post_angular_momentum,
        nlp_pre.states.mx,
        nlp_post.states.mx,
    )(pre_states_cx, post_states_cx)

    return func
