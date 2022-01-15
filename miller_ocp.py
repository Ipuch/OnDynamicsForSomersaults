import biorbd_casadi as biorbd
import numpy as np
from scipy import interpolate
from bioptim import (
    OdeSolver,
    Node,
    OptimalControlProgram,
    ConstraintFcn,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    ControlType,
    Bounds,
    Solver,
    InitialGuess,
    InterpolationType,
    PhaseTransitionList,
    BiMappingList,
)


class MillerOcp:
    def __init__(
            self,
            biorbd_model_path: str = None,
            n_shooting: int = 100,
            duration: float = 1.545,
            n_threads: int = 8,
            control_type: ControlType = ControlType.CONSTANT,
            ode_solver: OdeSolver = OdeSolver.COLLOCATION(),
            implicit_dynamics: bool = False,
            semi_implicit_dynamics: bool = False,
            vertical_velocity_0=9.2,  # Real data
            somersaults=4 * np.pi,
            twists=4 * np.pi,
    ):
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting
        self.duration = duration
        self.n_threads = n_threads
        self.control_type = control_type
        self.ode_solver = ode_solver
        self.implicit_dynamics = implicit_dynamics
        self.semi_implicit_dynamics = semi_implicit_dynamics

        self.vertical_velocity_0 = vertical_velocity_0
        self.somersaults = somersaults
        self.twists = twists
        self.somersault_rate_0 = somersaults / duration

        if biorbd_model_path is not None:
            self.biorbd_model = biorbd.Model(biorbd_model_path)

            self.n_q = self.biorbd_model.nbQ()
            self.n_qdot = self.biorbd_model.nbQdot()
            self.n_qddot = self.biorbd_model.nbQddot()
            self.n_tau = self.biorbd_model.nbGeneralizedTorque() - self.biorbd_model.nbRoot()

            self.tau_min, self.tau_init, self.tau_max = -200, 0, 200
            self.qddot_min, self.qddot_init, self.qddot_max = -1000, 0, 1000

            self.implicit_dynamics = implicit_dynamics
            self.mod = 2 if self.implicit_dynamics else 1

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()
            self.phase_transitions = PhaseTransitionList()
            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()
            self.initial_states = []
            self.x_init = InitialGuessList()
            self.u_init = InitialGuessList()

            self.control_type = control_type
            self.control_nodes = Node.ALL if self.control_type == ControlType.LINEAR_CONTINUOUS else Node.ALL_SHOOTING

            self._set_dynamics()
            self._set_constraints()
            self._set_objective_functions()

            self._set_boundary_conditions()
            self._set_initial_guesses()

            self.mapping = BiMappingList()
            self.mapping.add("tau", [None, None, None, None, None, None, 0, 1, 2, 3], [6, 7, 8, 9])

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                self.n_shooting,
                self.duration,
                x_init=self.x_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                constraints=self.constraints,
                n_threads=n_threads,
                variable_mappings=self.mapping,
                # control_type=self.control_type,
                ode_solver=ode_solver,
                use_sx=False,
            )

    def _set_dynamics(self):

        #  if XXX
        self.dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN, implicit_dynamics=self.implicit_dynamics, with_contact=False, phase=0
        )
        # elif floating base
        # self.dynamics.add(custom_configure, dynamic_function=root_explicit_dynamic)

    def _set_objective_functions(self):
        # --- Objective function --- #
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key="qdot", weight=5)  # Regularization
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, reference_jcs=3, marker_index=6,
            weight=1000)  # Right hand trajetory
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, reference_jcs=7, marker_index=11,
            weight=1000)  # Left hand trajectory

    def _set_constraints(self):
        # --- Constraints --- #
        # Set time as a variable
        slack_duration = 0.5
        self.constraints.add(ConstraintFcn.TIME_CONSTRAINT,
                             node=Node.END,
                             min_bound=self.duration - slack_duration,
                             max_bound=self.duration + slack_duration)

    def _set_initial_guesses(self):
        # --- Initial guess --- #
        # Initialize state vector
        self.x = np.zeros((self.n_q + self.n_qdot, self.n_shooting + 1))
        # data points
        data_point = np.linspace(0, self.duration, self.n_shooting + 1)

        # parabolic trajectory on Y
        self.x[2, :] = self.vertical_velocity_0 * data_point + -9.81 / 2 * data_point ** 2
        # Somersaults
        self.x[3, :] = np.linspace(0, self.somersaults, self.n_shooting + 1)
        # Twists
        self.x[5, :] = np.linspace(0, self.twists, self.n_shooting + 1)

        # Handle second DoF of arms with Noise.
        noise = np.random.random((1, self.n_shooting + 1))
        # self.xx[7, :] = noise * np.pi/2 - (np.pi - np.pi/4)
        # self.xx[9, :] = noise * np.pi/2 + np.pi/4
        self.x[7, :] = - np.pi / 2 + np.pi / 4 * np.sin(np.linspace(0, (2 * np.pi), self.n_shooting + 1) * self.duration * 2) \
                  + (noise - 0.5) * np.pi / 10
        self.x[9, :] = np.pi / 2 + np.pi / 4 * np.sin(np.linspace(0, (2 * np.pi), self.n_shooting + 1) * self.duration * 2) \
                  - (noise - 0.5) * np.pi / 10

        # velocity on Y
        self.x[self.n_q + 2, :] = self.vertical_velocity_0 - 9.81 * data_point
        # Somersaults rate
        self.x[self.n_q + 3, :] = self.somersault_rate_0
        # Twists rate
        self.x[self.n_q + 5, :] = self.twists / self.duration

        self._set_initial_states(self.x)
        self._set_initial_controls()

    def _set_initial_states(self, X0: np.array = None):
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

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None:
            if self.implicit_dynamics:
                self.u_init = InitialGuess([self.tau_init] * self.n_tau + [self.qddot_init] * self.n_qddot
                                           + [5] * self.biorbd_model.nbContacts())
            elif self.semi_implicit_dynamics:
                self.u_init = InitialGuess([self.tau_init] * self.n_tau + [self.qddot_init] * self.n_qddot)
            else:
                self.u_init = InitialGuess([self.tau_init] * self.n_tau)
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)
            self.u_init = InitialGuess(U0, interpolation=InterpolationType.EACH_FRAME)

    def _set_boundary_conditions(self):
        self.x_bounds = BoundsList()

        velocity_max = 100
        tilt_bound = np.pi / 4
        tilt_final_bound = np.pi / 12  # 15 degrees

        initial_arm_elevation = 2.8
        arm_rotation_z_upp = np.pi/2
        arm_rotation_z_low = 1
        arm_elevation_y_upp = np.pi

        slack_initial_vertical_velocity = 2
        slack_initial_somersault_rate = 1

        slack_somersault = 0.1
        slack_twist = 0.1

        slack_final_somersault = 0.1
        slack_final_twist = 0.1

        x_min = np.zeros((self.n_q + self.n_qdot, 3))
        x_max = np.zeros((self.n_q + self.n_qdot, 3))

        x_min[:self.n_q, 0] = [0, 0, 0, 0, 0, 0, 0, -initial_arm_elevation, 0, initial_arm_elevation]
        x_min[self.n_q:, 0] = [-1, -1, self.vertical_velocity_0 - slack_initial_vertical_velocity,
                               self.somersault_rate_0 - slack_initial_somersault_rate, 0, 0, 0, 0, 0, 0]

        x_max[:self.n_q, 0] = [0, 0, 0, 0, 0, 0, 0, -initial_arm_elevation, 0, initial_arm_elevation]
        x_max[self.n_q:, 0] = [-1, -1, self.vertical_velocity_0 + slack_initial_vertical_velocity,
                               self.somersault_rate_0 + slack_initial_somersault_rate, 0, 0, 0, 0, 0, 0]

        x_min[:self.n_q, 1] = [-3, -3, -0.001, -0.001, -tilt_bound, -0.001,
                               -arm_rotation_z_low, -arm_elevation_y_upp,
                               -arm_rotation_z_upp, 0]
        x_min[self.n_q:, 1] = - velocity_max

        x_max[:self.n_q, 1] = [3, 3, 10, self.somersaults + slack_somersault, tilt_bound, self.twists + slack_twist,
                               arm_rotation_z_upp, 0,
                               arm_rotation_z_low, arm_elevation_y_upp]
        x_max[self.n_q:, 1] = + velocity_max

        x_min[:self.n_q, 2] = [-0.1, -0.1, -0.1,
                               self.somersaults - slack_final_somersault, -tilt_final_bound, self.twists - slack_final_twist,
                               -arm_rotation_z_low, -arm_elevation_y_upp,
                               -arm_rotation_z_upp, 0]
        x_min[self.n_q:, 2] = - velocity_max

        x_max[:self.n_q, 2] = [0.1, 0.1, 0.1,
                               self.somersaults + slack_final_somersault, tilt_final_bound, self.twists + slack_final_twist,
                               arm_rotation_z_upp, 0,
                               arm_rotation_z_low, arm_elevation_y_upp]
        x_max[self.n_q:, 2] = + velocity_max

        self.x_bounds.add(
            bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

        if self.implicit_dynamics:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot
                + [self.qddot_min] * self.biorbd_model.nbContacts(),
                [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot
                + [self.qddot_max] * self.biorbd_model.nbContacts(),
            )
        elif self.semi_implicit_dynamics:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
                [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
            )
        else:
            self.u_bounds.add([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau)

    def _interpolate_initial_states(self, X0: np.array):
        print("interpolating initial states to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, X0.shape[1])
        y = X0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting + 1)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

    def _interpolate_initial_controls(self, U0: np.array):
        print("interpolating initial controls to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, U0.shape[1])
        y = U0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

