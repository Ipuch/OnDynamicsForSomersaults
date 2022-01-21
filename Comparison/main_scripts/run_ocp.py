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


def solve(
    n_shooting_per_second: int = 50,
    model_path: str = None,
    ode_solver: OdeSolver = OdeSolver.RK8(),
    tol: float = 1e-8,
    online_optim: bool = False,
    X0: Any = None,
    U0: Any = None,
):

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path=model_path,
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
