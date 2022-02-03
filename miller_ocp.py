import biorbd_casadi as biorbd
import numpy as np
from scipy import interpolate
from typing import Union
import casadi as cas
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
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    PenaltyNodeList,
    BiorbdInterface,
)

from custom_dynamics.root_explicit_qddot_joint import root_explicit_dynamic, custom_configure_root_explicit
from custom_dynamics.root_implicit import root_implicit_dynamic, custom_configure_root_implicit


class MillerOcp:
    def __init__(
        self,
        biorbd_model_path: str = None,
        n_shooting: tuple = (125, 25),
        duration: float = 1.545,
        n_threads: int = 8,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        dynamics_type: str = "explicit",
        vertical_velocity_0: float = 9.2,  # Real data
        somersaults: float = 4 * np.pi,
        twists: float = 6 * np.pi,
        use_sx: bool = False,
    ):
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting
        self.duration = duration
        self.n_threads = n_threads
        self.ode_solver = ode_solver
        self.dynamics_type = dynamics_type

        self.vertical_velocity_0 = vertical_velocity_0
        self.somersaults = somersaults
        self.twists = twists
        self.somersault_rate_0 = somersaults / duration

        if biorbd_model_path is not None:
            self.biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))
            self.dynamics_type = dynamics_type

            self.n_q = self.biorbd_model[0].nbQ()
            self.n_qdot = self.biorbd_model[0].nbQdot()
            self.nb_root = self.biorbd_model[0].nbRoot()

            if self.dynamics_type == "implicit" or self.dynamics_type == "root_implicit":
                self.n_qddot = self.biorbd_model[0].nbQddot()
            elif self.dynamics_type == "explicit" or self.dynamics_type == "root_explicit":
                self.n_qddot = self.biorbd_model[0].nbQddot() - self.biorbd_model[0].nbRoot()

            self.n_tau = self.biorbd_model[0].nbGeneralizedTorque() - self.biorbd_model[0].nbRoot()

            self.tau_min, self.tau_init, self.tau_max = -100, 0, 100
            self.qddot_min, self.qddot_init, self.qddot_max = -1000, 0, 1000

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()
            self.phase_transitions = PhaseTransitionList()
            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()
            self.initial_states = []
            self.x_init = InitialGuessList()
            self.u_init = InitialGuessList()
            self.mapping = BiMappingList()

            self._set_dynamics()
            # self._set_constraints()
            self._set_objective_functions()

            self._set_boundary_conditions()
            self._set_initial_guesses()

            self._set_mapping()

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                self.n_shooting,
                (7 / 8 * self.duration, 1 / 8 * self.duration),
                x_init=self.x_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                # constraints=self.constraints,
                n_threads=n_threads,
                variable_mappings=self.mapping,
                control_type=ControlType.CONSTANT,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _set_dynamics(self):
        for phase in range(len(self.n_shooting)):
            if self.dynamics_type == "explicit":
                self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
            elif self.dynamics_type == "root_explicit":
                self.dynamics.add(custom_configure_root_explicit, dynamic_function=root_explicit_dynamic)
            elif self.dynamics_type == "implicit":
                self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, implicit_dynamics=True, with_contact=False)
            elif self.dynamics_type == "root_implicit":
                self.dynamics.add(custom_configure_root_implicit, dynamic_function=root_implicit_dynamic)
            else:
                raise ValueError("Check spelling, choices are explicit, root_explicit, implicit, root_implicit")

    def _set_objective_functions(self):
        def custom_angular_momentum(all_pn: PenaltyNodeList) -> cas.MX:
            angular_momentum = BiorbdInterface.mx_to_cx(
                "angularMomentum", all_pn.nlp.model.angularMomentum, all_pn.nlp.states["q"], all_pn.nlp.states["qdot"]
            )
            return angular_momentum

        # --- Objective function --- #
        for i in range(len(self.n_shooting)):
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                derivative=True,
                key="qdot",
                index=(6, 7, 8, 9, 10, 11, 12, 13, 14),
                weight=1,
                phase=i,
            )  # Regularization
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=1,
                marker_index=6,
                weight=10,
                phase=i,
            )  # Right hand trajetory
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=1,
                marker_index=11,
                weight=10,
                phase=i,
            )  # Left hand trajectory
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=0,
                marker_index=16,
                weight=100000,
                phase=i,
            )  # feet trajectory
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=(6, 7, 8, 13, 14), key="q", weight=10, phase=i
            )  # core DoFs

        self.objective_functions.add(
            custom_angular_momentum, custom_type=ObjectiveFcn.Mayer, node=Node.START, weight=100000
        )

        slack_duration = 0.3
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=7 / 8 * self.duration - 1 / 2 * slack_duration,
            max_bound=7 / 8 * self.duration + 1 / 2 * slack_duration,
            phase=0,
            weight=1e-6,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=1 / 8 * self.duration - 1 / 2 * slack_duration,
            max_bound=1 / 8 * self.duration + 1 / 2 * slack_duration,
            phase=1,
            weight=1e-6,
        )

    # def _set_constraints(self):
    #     --- Constraints --- #
    #     Set time as a variable
    #     slack_duration = 0.3
    #     self.constraints.add(ConstraintFcn.TIME_CONSTRAINT,
    #                          node=Node.END,
    #                          min_bound=7/8 * self.duration - 1/2*slack_duration,
    #                          max_bound=7/8 * self.duration + 1/2*slack_duration, phase=0)
    #     self.constraints.add(ConstraintFcn.TIME_CONSTRAINT,
    #                          node=Node.END,
    #                          min_bound=1/8 * self.duration - 1/2*slack_duration,
    #                          max_bound=1/8 * self.duration + 1/2*slack_duration, phase=1)

    def _set_initial_guesses(self):
        # --- Initial guess --- #
        # Initialize state vector
        self.x = np.zeros((self.n_q + self.n_qdot, np.sum(self.n_shooting) + len(self.n_shooting)))

        # data points
        data_point = np.linspace(0, self.duration, np.sum(self.n_shooting) + len(self.n_shooting))

        # parabolic trajectory on Y
        self.x[2, :] = self.vertical_velocity_0 * data_point + -9.81 / 2 * data_point**2
        # Somersaults
        self.x[3, :] = np.linspace(0, self.somersaults, np.sum(self.n_shooting) + len(self.n_shooting))
        # Twists
        self.x[5, :] = np.linspace(0, self.twists, np.sum(self.n_shooting) + len(self.n_shooting))

        # Handle second DoF of arms with Noise.
        self.x[6:9, :] = np.random.random((3, np.sum(self.n_shooting) + len(self.n_shooting))) * np.pi / 12 - np.pi / 24
        self.x[10, :] = np.random.random((1, np.sum(self.n_shooting) + len(self.n_shooting))) * np.pi / 2 - (
            np.pi - np.pi / 4
        )
        self.x[12, :] = np.random.random((1, np.sum(self.n_shooting) + len(self.n_shooting))) * np.pi / 2 + np.pi / 4
        self.x[13:15, :] = (
            np.random.random((2, np.sum(self.n_shooting) + len(self.n_shooting))) * np.pi / 12 - np.pi / 24
        )

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
            self.x_init.add([0] * (self.n_q + self.n_q))
        else:
            if X0.shape[1] != np.sum(self.n_shooting) + len(self.n_shooting):
                X0 = self._interpolate_initial_states(X0)

            if self.ode_solver.is_direct_shooting:
                shooting = 0
                for i in range(len(self.n_shooting)):
                    self.x_init.add(
                        X0[:, shooting : shooting + self.n_shooting[i] + 1], interpolation=InterpolationType.EACH_FRAME
                    )
                    shooting += self.n_shooting[i]
            else:
                n = self.ode_solver.polynomial_degree
                X0 = np.repeat(X0, n + 1, axis=1)
                X0 = X0[:, :-n]
                self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None:
            for phase in range(len(self.n_shooting)):
                if self.dynamics_type == "explicit":
                    self.u_init.add([self.tau_init] * self.n_tau)
                elif self.dynamics_type == "root_explicit":
                    self.u_init.add([self.qddot_init] * self.n_qddot)
                elif self.dynamics_type == "implicit":
                    self.u_init.add([self.tau_init] * self.n_tau + [self.qddot_init] * self.n_qddot)
                elif self.dynamics_type == "root_implicit":
                    self.u_init.add([self.qddot_init] * self.n_qddot)
                else:
                    raise ValueError("Check spelling, choices are explicit, root_explicit, implicit, root_implicit")
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)

                shooting = 0
                for i in range(len(self.n_shooting)):
                    self.u_init.add(
                        U0[:, shooting : shooting + self.n_shooting[i]], interpolation=InterpolationType.EACH_FRAME
                    )
                    shooting += self.n_shooting[i]

    def _set_boundary_conditions(self):
        self.x_bounds = BoundsList()

        velocity_max = 100
        velocity_max_phase_transition = 10
        tilt_bound = np.pi / 4
        tilt_final_bound = np.pi / 12  # 15 degrees

        initial_arm_elevation = 2.8
        arm_rotation_z_upp = np.pi / 2
        arm_rotation_z_low = 1
        arm_elevation_y_low = 0.01
        arm_elevation_y_upp = np.pi - 0.01
        thorax_hips_xyz = np.pi / 6
        arm_rotation_y_final = 2.4

        slack_initial_vertical_velocity = 2
        slack_initial_somersault_rate = 3

        slack_somersault = 0.5
        slack_twist = 0.5

        slack_final_somersault = np.pi / 24  # 7.5 degrees
        slack_final_twist = np.pi / 24  # 7.5 degrees
        slack_final_dofs = np.pi / 24  # 7.5 degrees

        x_min = np.zeros((2, self.n_q + self.n_qdot, 3))
        x_max = np.zeros((2, self.n_q + self.n_qdot, 3))

        x_min[0, : self.n_q, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -initial_arm_elevation, 0, initial_arm_elevation, 0, 0]
        x_min[0, self.n_q :, 0] = [
            -1,
            -1,
            self.vertical_velocity_0 - slack_initial_vertical_velocity,
            self.somersault_rate_0 - slack_initial_somersault_rate,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        x_max[0, : self.n_q, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -initial_arm_elevation, 0, initial_arm_elevation, 0, 0]
        x_max[0, self.n_q :, 0] = [
            1,
            1,
            self.vertical_velocity_0 + slack_initial_vertical_velocity,
            self.somersault_rate_0 + slack_initial_somersault_rate,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        x_min[0, : self.n_q, 1] = [
            -3,
            -3,
            -0.001,
            -0.001,
            -tilt_bound,
            -0.001,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
            -arm_rotation_z_low,
            -arm_elevation_y_upp,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
        ]
        x_min[0, self.n_q :, 1] = -velocity_max

        x_max[0, : self.n_q, 1] = [
            3,
            3,
            10,
            self.somersaults + slack_somersault,
            tilt_bound,
            self.twists + slack_twist,
            thorax_hips_xyz,
            thorax_hips_xyz,
            thorax_hips_xyz,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            arm_elevation_y_upp,
            thorax_hips_xyz,
            thorax_hips_xyz,
        ]
        x_max[0, self.n_q :, 1] = +velocity_max

        x_min[0, : self.n_q, 2] = [
            -3,
            -3,
            -0.001,
            7 / 8 * self.somersaults - slack_final_somersault,
            -tilt_final_bound,
            self.twists - slack_twist,
            -slack_final_dofs,
            -slack_final_dofs,
            -slack_final_dofs,
            -arm_rotation_z_low,
            -0.2,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            thorax_hips_xyz - slack_final_dofs,
            -slack_final_dofs,
        ]  # x_min[0, :self.n_q, 1]
        x_min[0, self.n_q :, 2] = -velocity_max

        x_max[0, : self.n_q, 2] = [
            3,
            3,
            10,
            7 / 8 * self.somersaults + slack_final_somersault,
            tilt_final_bound,
            self.twists + slack_twist,
            slack_final_dofs,
            slack_final_dofs,
            slack_final_dofs,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            0.2,
            thorax_hips_xyz,
            slack_final_dofs,
        ]  # x_max[0, :self.n_q, 1]
        x_max[0, self.n_q :, 2] = +velocity_max

        x_min[1, : self.n_q, 0] = x_min[0, : self.n_q, 2]
        x_min[1, self.n_q :, 0] = x_min[0, self.n_q :, 2]

        x_max[1, : self.n_q, 0] = x_max[0, : self.n_q, 2]
        x_max[1, self.n_q :, 0] = x_max[0, self.n_q :, 2]

        x_min[1, : self.n_q, 1] = [
            -3,
            -3,
            -0.001,
            7 / 8 * self.somersaults - slack_final_somersault,
            -tilt_bound,
            self.twists - slack_final_twist,
            -slack_final_dofs,
            -slack_final_dofs,
            -slack_final_dofs,
            -arm_rotation_z_low,
            -arm_elevation_y_upp,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            -slack_final_dofs,
            -slack_final_dofs,
        ]  # x_min[0, :self.n_q, 1]
        x_min[1, self.n_q :, 1] = [
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max_phase_transition,
        ]

        x_max[1, : self.n_q, 1] = [
            3,
            3,
            10,
            self.somersaults + slack_somersault,
            tilt_bound,
            self.twists + slack_twist,
            slack_final_dofs,
            slack_final_dofs,
            slack_final_dofs,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            arm_elevation_y_upp,
            thorax_hips_xyz,
            slack_final_dofs,
        ]  #  x_max[0, :self.n_q, 1]
        x_max[1, self.n_q :, 1] = [
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max_phase_transition,
        ]

        x_min[1, : self.n_q, 2] = [
            -0.1,
            -0.1,
            -0.1,
            self.somersaults - thorax_hips_xyz - slack_final_somersault,
            -tilt_final_bound,
            self.twists - slack_final_twist,
            -slack_final_dofs,
            -slack_final_dofs,
            -slack_final_dofs,
            -arm_rotation_z_low,
            -arm_elevation_y_upp,
            -arm_rotation_z_upp,
            arm_rotation_y_final,
            thorax_hips_xyz - slack_final_dofs,
            -slack_final_dofs,
        ]
        x_min[1, self.n_q :, 2] = [
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max_phase_transition,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max,
            -velocity_max_phase_transition,
        ]

        x_max[1, : self.n_q, 2] = [
            0.1,
            0.1,
            0.1,
            self.somersaults - thorax_hips_xyz,
            tilt_final_bound,
            self.twists + slack_final_twist,
            slack_final_dofs,
            slack_final_dofs,
            slack_final_dofs,
            arm_rotation_z_upp,
            -arm_rotation_y_final,
            arm_rotation_z_low,
            arm_elevation_y_upp,
            thorax_hips_xyz,
            slack_final_dofs,
        ]
        x_max[1, self.n_q :, 2] = [
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max_phase_transition,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max,
            velocity_max_phase_transition,
        ]

        for phase in range(len(self.n_shooting)):
            self.x_bounds.add(
                bounds=Bounds(
                    x_min[phase, :, :],
                    x_max[phase, :, :],
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )
            )

            if self.dynamics_type == "explicit":
                self.u_bounds.add([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau)
            elif self.dynamics_type == "root_explicit":
                self.u_bounds.add([self.qddot_min] * self.n_qddot, [self.qddot_max] * self.n_qddot)
            elif self.dynamics_type == "implicit":
                self.u_bounds.add(
                    [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
                    [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
                )
            elif self.dynamics_type == "root_implicit":
                self.u_bounds.add([self.qddot_min] * self.n_qddot, [self.qddot_max] * self.n_qddot)
            else:
                raise ValueError("Check spelling, choices are explicit, root_explicit, implicit, root_implicit")

    def _interpolate_initial_states(self, X0: np.array):
        print("interpolating initial states to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, X0.shape[1])
        y = X0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, np.sum(self.n_shooting) + len(self.n_shooting))
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

    def _set_mapping(self):
        if self.dynamics_type == "explicit":
            self.mapping.add(
                "tau", [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 10, 11, 12, 13, 14]
            )
        elif self.dynamics_type == "root_explicit":
            print("no bimapping")
            # self.mapping.add(
            #     "qddot",
            #     [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            #     [6, 7, 8, 9, 10, 11, 12, 13, 14],
            # )
        elif self.dynamics_type == "implicit":
            self.mapping.add(
                "tau", [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 10, 11, 12, 13, 14]
            )
        elif self.dynamics_type == "root_implicit":
            pass
            # self.mapping.add("qddot", [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 10, 11, 12, 13, 14])
        else:
            raise ValueError("Check spelling, choices are explicit, root_explicit, implicit, root_implicit")
