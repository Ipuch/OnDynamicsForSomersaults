from bioptim import PhaseTransition, NonLinearProgram, MultinodeConstraint
from casadi import MX, vertcat
import biorbd_casadi as biorbd
from typing import Union


def minimize_angular_momentum(
    transition: Union[PhaseTransition, MultinodeConstraint],
    nlp_pre: NonLinearProgram,
    nlp_post: NonLinearProgram,
) -> MX:
    """
    The constraint of the transition which maintain the difference between angular momentum to zero.

    Parameters
    ----------
    transition: Union[PhaseTransition, MultinodeConstraint]
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
    # q_post = nlp_post.states["q"].mx

    q_post = MX.sym("q", *nlp_post.states["q"].mx.shape)

    qdot_pre = nlp_pre.states["qdot"].mx
    # qdot_post = nlp_post.states["qdot"].mx
    qdot_post = MX.sym("qdot", *nlp_post.states["qdot"].mx.shape)

    pre_angular_momentum = nlp_pre.model.angularMomentum(q_pre, qdot_pre).to_mx()
    post_angular_momentum = nlp_post.model.angularMomentum(q_post, qdot_post).to_mx()

    pre_states_cx = nlp_pre.states.cx_end
    post_states_cx = nlp_post.states.cx

    x_post = vertcat(q_post, qdot_post)

    func = biorbd.to_casadi_func(
        "angular_momentum_transition",
        pre_angular_momentum - post_angular_momentum,
        nlp_pre.states.mx,
        x_post,
    )(pre_states_cx, post_states_cx)

    return func


def minimize_linear_momentum(
    transition: Union[PhaseTransition, MultinodeConstraint],
    nlp_pre: NonLinearProgram,
    nlp_post: NonLinearProgram,
) -> MX:
    """
    The constraint of the transition which maintain the difference between angular momentum to zero.

    Parameters
    ----------
    transition: Union[PhaseTransition, MultinodeConstraint]
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
    # q_post = nlp_post.states["q"].mx

    q_post = MX.sym("q", *nlp_post.states["q"].mx.shape)

    qdot_pre = nlp_pre.states["qdot"].mx
    # qdot_post = nlp_post.states["qdot"].mx
    qdot_post = MX.sym("qdot", *nlp_post.states["qdot"].mx.shape)

    pre_linear_momentum = nlp_pre.model.mass().to_mx() * nlp_pre.model.CoMdot(q_pre, qdot_pre).to_mx()
    post_linear_momentum = nlp_pre.model.mass().to_mx() * nlp_post.model.CoMdot(q_post, qdot_post).to_mx()

    pre_states_cx = nlp_pre.states.cx_end
    post_states_cx = nlp_post.states.cx

    x_post = vertcat(q_post, qdot_post)

    func = biorbd.to_casadi_func(
        "linear_momentum_transition",
        pre_linear_momentum - post_linear_momentum,
        nlp_pre.states.mx,
        x_post,
    )(pre_states_cx, post_states_cx)

    return func
