"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    CostType,
    Shooting,
    Solver,
    SolutionIntegrator,
    InterpolationType,
)
from typing import Any, Union


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    n_shooting_per_second: int = 30,
    use_sx: bool = True,
    n_threads: int = 1,
    X0: Any = None,
    U0: Any = None,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting_per_second: int
        The number of shooting points to define int the direct multiple shooting program by second
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    X0: np.array
    U0: np.array

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    n_shooting = n_shooting_per_second * final_time

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, 0] = 0
    x_bounds[:, -1] = 0
    x_bounds[0, -1] = 3.14

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    if X0 is None:
        x_init = InitialGuess([0] * (n_q + n_qdot))
    else:
        x_init = InitialGuess(X0, interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)

    if U0 is None:
        u_init = InitialGuess([tau_init] * n_tau)
    else:
        u_init = InitialGuess(U0, interpolation=InterpolationType.EACH_FRAME)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=n_threads,
    )


def main(
    n_shooting_per_second: int = 50,
    ode_solver: OdeSolver = OdeSolver.RK8(),
    tol: float = 1e-8,
    online_optim: bool = False,
    X0: Any = None,
    U0: Any = None,
):

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path="models/triple_pendulum.bioMod",
        final_time=1,
        n_shooting_per_second=n_shooting_per_second,
        ode_solver=ode_solver,
        X0=X0,
        U0=U0,
    )

    # --- Solve the ocp --- #
    options = Solver.IPOPT(show_online_optim=online_optim)
    options.set_convergence_tolerance(tol)
    options.set_constraint_tolerance(tol)
    options.set_maximum_iterations(2000)
    options.set_limited_memory_max_history(50)
    options.set_linear_solver("mumps")
    options.set_print_level(0)

    sol = ocp.solve(options)

    return ocp, sol


def integrate_sol(ocp, sol):
    sol_int = sol.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS, use_scipy_integrator=True, keep_intermediate_points=False
    )
    return sol_int.states["all"]


def compute_error_single_shooting(sol, duration: float = 1, integrator: str = None):
    sol_merged = sol.merge_phases()

    if sol_merged.phase_time[-1] < duration:
        raise ValueError(
            f"Single shooting integration duration must be smaller than ocp duration :{sol_merged.phase_time[-1]} s"
        )

    trans_idx = []
    rot_idx = []
    for i in range(sol.ocp.nlp[0].model.nbQ()):
        if sol.ocp.nlp[0].model.nameDof()[i].to_string()[-4:-1] == "Rot":
            rot_idx += [i]
        else:
            trans_idx += [i]
    rot_idx = np.array(rot_idx)
    trans_idx = np.array(trans_idx)

    sol_int = sol.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        merge_phases=True,
        integrator=integrator,
        keep_intermediate_points=False,
    )
    sn_1s = int(sol_int.ns[0] / sol_int.phase_time[-1] * duration)  # shooting node at {duration} second
    if len(rot_idx) > 0:
        single_shoot_error_r = (
            np.sqrt(np.mean((sol_int.states["q"][rot_idx, sn_1s] - sol_merged.states["q"][rot_idx, sn_1s]) ** 2))
            * 180
            / np.pi
        )
    else:
        single_shoot_error_r = "N.A."
    if len(trans_idx) > 0:
        single_shoot_error_t = (
            np.sqrt(
                np.mean((sol_int.states["q"][trans_idx, 5 * sn_1s] - sol_merged.states["q"][trans_idx, sn_1s]) ** 2)
            )
            / 1000
        )
    else:
        single_shoot_error_t = "N.A."
    return single_shoot_error_t, single_shoot_error_r


if __name__ == "__main__":
    T = np.zeros((6, 3))

    ocp, sol = main(ode_solver=OdeSolver.RK8())
    print(OdeSolver.RK8().rk_integrator.__name__)
    for cpt, i in enumerate(SolutionIntegrator):
        print(i.value)
        print(compute_error_single_shooting(sol, duration=1, integrator=i))
        a, b = compute_error_single_shooting(sol, duration=1, integrator=i)
        T[cpt, 0] = b

    # sol.print()
    # sol.animate()

    ocp, sol = main(ode_solver=OdeSolver.IRK(method="legendre", polynomial_degree=5))
    # # sol.print()
    # # sol.graphs(integrator=SolutionIntegrator.SCIPY_RK23)
    # # sol.animate()
    print(OdeSolver.IRK().rk_integrator.__name__)
    for cpt, i in enumerate(SolutionIntegrator):
        print(i.value)
        print(compute_error_single_shooting(sol, duration=1, integrator=i))
        a, b = compute_error_single_shooting(sol, duration=1, integrator=i)
        T[cpt, 1] = b

    ocp, sol = main(ode_solver=OdeSolver.CVODES())
    # sol.print()
    # sol.graphs(integrator=SolutionIntegrator.SCIPY_RK23)
    # sol.animate()
    print(OdeSolver.CVODES().rk_integrator.__name__)
    for cpt, i in enumerate(SolutionIntegrator):
        print(i.value)
        print(compute_error_single_shooting(sol, duration=1, integrator=i))
        a, b = compute_error_single_shooting(sol, duration=1, integrator=i)
        T[cpt, 2] = b

    np.savetxt("table.txt", T)
