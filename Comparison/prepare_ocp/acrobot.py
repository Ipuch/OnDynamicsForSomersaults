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
    ObjectiveList,
    ConstraintList,
)
from typing import Any, Union
from scipy import interpolate


class AcrobotOCP:
    def __init__(
        self,
        model_path: str,
        solver: Solver = Solver.IPOPT,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        tol: float = 1e-8,
        n_threads: int = 8,
        n_shooting_per_second: int = 100,
        final_time: float = 1.0,
        X0: Any = None,
        U0: Any = None,
        online_optim: bool = False,
    ):

        self.model_path = model_path
        self.model = biorbd.Model(self.model_path)
        self.n_q = self.model.nbQ()
        self.n_tau = self.model.nbGeneralizedTorque()

        self.tau_min = -300
        self.tau_max = 300
        self.tau_init = 0

        self.n_shooting_per_second = n_shooting_per_second
        self.time = final_time
        self.n_shooting = int(self.n_shooting_per_second * self.time)

        self.ode_solver = ode_solver
        self.solver = solver
        self.n_threads = n_threads

        self.tol = tol
        self.online_optim = online_optim

        self.dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

        self.x_bounds = Bounds()
        self.u_bounds = Bounds()
        self._set_bounds()

        self.x_init = InitialGuess()
        self.u_init = InitialGuess()
        self._set_initial_states(X0)
        self._set_initial_controls(U0)

        self.objective_functions = ObjectiveList()
        self._set_generic_objective_functions()

        self.constraints = ConstraintList()
        self._set_generic_constraints()

        self._set_generic_ocp()

    def _set_generic_ocp(self):
        """
        The initialization of an ocp

        Returns
        -------
        The OptimalControlProgram ready to be solved
        """
        self.ocp = OptimalControlProgram(
            self.model,
            self.dynamics,
            n_shooting=self.n_shooting,
            phase_time=self.time,
            x_init=self.x_init,
            u_init=self.u_init,
            x_bounds=self.x_bounds,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            ode_solver=self.ode_solver,
            use_sx=False,
            n_threads=self.n_threads,
        )

    def solve(self):

        # --- Prepare the ocp --- #

        # --- Solve the ocp --- #
        options = Solver.IPOPT(show_online_optim=self.online_optim, show_options={"show_bounds": True})
        options.set_convergence_tolerance(self.tol)
        options.set_constraint_tolerance(self.tol)
        options.set_maximum_iterations(1)
        # options.set_limited_memory_max_history(50)
        options.set_linear_solver("mumps")
        options.set_print_level(4)

        self.ocp.add_plot_penalty(CostType.ALL)

        self.sol = self.ocp.solve(options)

        return self.ocp, self.sol

    def _set_bounds(self):
        self.x_bounds = QAndQDotBounds(self.model)
        # self.x_bounds[:, 0] = 0
        self.x_bounds.min[self.n_q :, :] = -3.14 * 100
        self.x_bounds.max[self.n_q :, :] = 3.14 * 100
        self.x_bounds.min[self.n_q :, 0] = -1e-2
        self.x_bounds.max[self.n_q :, 0] = 1e-2

        # self.x_bounds[0, 0] = 3.14
        self.x_bounds[0, 0] = np.pi / 2
        self.x_bounds[0, -1] = 3.14
        self.x_bounds.min[1, 0] = -1e-2
        self.x_bounds.max[1, 0] = 1e-2
        self.x_bounds.min[1, -1] = -1e-2
        self.x_bounds.max[1, -1] = 1e-2

        u_min = [self.tau_min] * self.n_tau
        u_max = [self.tau_max] * self.n_tau
        self.u_bounds = Bounds(u_min, u_max)
        self.u_bounds[0, :] = 0

    def _set_initial_states(self, X0):

        if X0 is None:
            self.x_init = InitialGuess([0] * (self.n_q + self.n_q))
        else:
            if X0.shape[1] != self.n_shooting + 1:
                X0 = self._interpolate_initial_states(X0)

            if self.ode_solver.is_direct_shooting:
                self.x_init = InitialGuess(X0, interpolation=InterpolationType.EACH_FRAME)
            else:
                n = self.ode_solver.polynomial_degree
                X0 = np.repeat(X0, n + 1, axis=1)
                X0 = X0[:, :-n]
                self.x_init = InitialGuess(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0):

        if U0 is None:
            self.u_init = InitialGuess([self.tau_init] * self.n_tau)
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)
            self.u_init = InitialGuess(U0, interpolation=InterpolationType.EACH_FRAME)

    def _set_generic_constraints(self):
        pass

    def _set_generic_objective_functions(self):
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
        # self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=1)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.1)

    def _interpolate_initial_states(self, X0):
        print("interpolating initial states to match the number of shooting nodes")
        x = np.linspace(0, self.time, X0.shape[1])
        y = X0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.time, self.n_shooting + 1)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

    def _interpolate_initial_controls(self, U0):
        print("interpolating initial controls to match the number of shooting nodes")
        x = np.linspace(0, self.time, U0.shape[1])
        y = U0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.time, self.n_shooting)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new
