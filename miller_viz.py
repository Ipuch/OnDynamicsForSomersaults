import numpy as np
import biorbd_casadi as biorbd


def plot_linear_momentum(x, nlp):
    linear_momentum_func = nlp.model.mass() * biorbd.to_casadi_func("LinearMomentum",  nlp.model.CoMdot,
                                                 nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False)

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.array(linear_momentum_func(q, qdot))


def plot_angular_momentum(x, nlp):
    angular_momentum_func = biorbd.to_casadi_func("AngularMomentum", nlp.model.angularMomentum,
                                                  nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False)

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.array(angular_momentum_func(q, qdot))


def plot_residual_torque(x, nlp):
    ID_func = biorbd.to_casadi_func("ResidualTorque", nlp.model.InverseDynamics,
                                                  nlp.states["q"].mx, nlp.states["qdot"].mx, nlp.controls["qddot"].mx, expand=True)

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])
    qddot = nlp.controls["qddot"]

    return np.array(ID_func(q, qdot, qddot)[:6])


def add_custom_plots(ocp, dynamics_type):
    for i, nlp in enumerate(ocp.nlp):
        # ocp.add_plot(
        #     "LinearMomentum", lambda t, x, u, p: plot_linear_momentum(x, nlp), phase=i,
        #     legend=["Linear Momentum x", "Linear Momentum y", "Linear Momentum z"]
        # )
        ocp.add_plot(
            "AngularMomentum", lambda t, x, u, p: plot_angular_momentum(x, nlp), phase=i,
            legend=["Angular Momentum x", "Angular Momentum y", "Angular Momentum z"]
        )
        if "implicit" in dynamics_type:
            ocp.add_plot(
                "AngularMomentum", lambda t, x, u, p: plot_residual_torque(x, nlp), phase=i,
                legend=["Angular Momentum x", "Angular Momentum y", "Angular Momentum z"]
            )
