"""
This file contains functions to plot extra information for the optimal control problem of the Miller.
"""

import numpy as np
import biorbd_casadi as biorbd
from bioptim import PlotType, NonLinearProgram, OptimalControlProgram
from custom_dynamics.enums import MillerDynamics


def plot_linear_momentum(x, nlp: NonLinearProgram):
    """
    Compute the linear momentum of the system.

    Parameters
    ----------
    x
        State vector
    nlp: NonLinearProgram
        Non linear program
    """
    linear_momentum = (
        nlp.model.mass().to_mx() * nlp.model.CoMdot(nlp.states["q"].mx, nlp.states["qdot"].mx, True).to_mx()
    )

    linear_momentum_func = biorbd.to_casadi_func(
        "LinearMomentum", linear_momentum, nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False
    )

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.array(linear_momentum_func(q, qdot))


def plot_angular_momentum(x, nlp: NonLinearProgram):
    """
    Compute the angular momentum of the system.

    Parameters
    ----------
    x
        State vector
    nlp: NonLinearProgram
        Non linear program
    """
    angular_momentum_func = biorbd.to_casadi_func(
        "AngularMomentum", nlp.model.angularMomentum, nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False
    )

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.array(angular_momentum_func(q, qdot))


def plot_residual_torque(x, u, nlp: NonLinearProgram):
    """
    Compute the residual torque of the system.

    Parameters
    ----------
    x
        State vector
    u
        Control vector
    nlp: NonLinearProgram
        Non linear program
    """
    qddot_mx = nlp.controls["qddot"].mx if "qddot" in nlp.controls.keys() else nlp.states["qddot"].mx
    ID_func = biorbd.to_casadi_func(
        "ResidualTorque",
        nlp.model.InverseDynamics,
        nlp.states["q"].mx,
        nlp.states["qdot"].mx,
        qddot_mx,
        expand=True,
    )

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])
    qddot = (
        u[nlp.controls["qddot"].index, :]
        if "qddot" in nlp.controls.keys()
        else nlp.states["qddot"].mapping.to_second.map(x[nlp.states["qddot"].index, :])
    )

    return np.array(ID_func(q, qdot, qddot)[:6, :])


def add_custom_plots(ocp: OptimalControlProgram, dynamics_type: MillerDynamics):
    """
    Add extra plots to the OCP.

    Parameters
    ----------
    ocp: OptimalControlProgram
        Optimal control program
    dynamics_type: MillerDynamics
        Type of dynamics of the Miller optimal control problem
    """
    for i, nlp in enumerate(ocp.nlp):

        ocp.add_plot(
            "LinearMomentum",
            lambda t, x, u, p: plot_linear_momentum(x, nlp),
            phase=i,
            legend=["Linear Momentum x", "Linear Momentum y", "Linear Momentum z"],
            plot_type=PlotType.PLOT,
        )

        ocp.add_plot(
            "AngularMomentum",
            lambda t, x, u, p: plot_angular_momentum(x, nlp),
            phase=i,
            legend=["Angular Momentum x", "Angular Momentum y", "Angular Momentum z"],
            plot_type=PlotType.PLOT,
        )
        if "implicit" in dynamics_type.value:
            ocp.add_plot(
                "TorqueResiduals",
                lambda t, x, u, p: plot_residual_torque(x, u, nlp),
                phase=i,
                legend=[
                    "TorqueResiduals x",
                    "TorqueResiduals y",
                    "TorqueResiduals z",
                    "TorqueResiduals Rx",
                    "TorqueResiduals Ry",
                    "TorqueResiduals Rz",
                ],
                plot_type=PlotType.PLOT,
            )
