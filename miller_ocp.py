import biorbd_casadi as biorbd
import biorbd as brd
import numpy as np
from scipy import interpolate
import pickle
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
    PhaseTransition,
    OptimizationVariableList,
)

from casadi import MX
from custom_dynamics.root_explicit_qddot_joint import root_explicit_dynamic, custom_configure_root_explicit
from custom_dynamics.root_implicit import root_implicit_dynamic, custom_configure_root_implicit


class MillerOcp:
    def __init__(
            self,
            biorbd_model_path: str = None,
            n_shooting: tuple = (125, 25),
            phase_durations: tuple = (1.501874663121631, 0.19569489882099642),  # t_tot = 1.545 c'était avant
            n_threads: int = 8,
            ode_solver: OdeSolver = OdeSolver.RK4(),
            dynamics_type: str = "explicit",
            vertical_velocity_0: float = 8.401410312445972,  # Real data 9.2 before
            somersaults: float = 4 * np.pi,
            twists: float = 6 * np.pi,
            use_sx: bool = False,
    ):
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting
        self.n_phases = len(n_shooting)

        # test to reload a previous solution.
        self.x = None
        self.u = None
        # self._load_initial_guess()
        self.phase_durations = phase_durations
        self.duration = np.sum(self.phase_durations)
        self.phase_proportions = (self.phase_durations[0] / self.duration,
                                  self.phase_durations[1] / self.duration)

        self.velocity_x = -0.006024447069071293
        self.velocity_y = 0.44094157867162764
        self.vertical_velocity_0 = vertical_velocity_0
        self.somersault_rate_0 = 5.133573161842047

        self.n_threads = n_threads
        self.ode_solver = ode_solver
        self.dynamics_type = dynamics_type

        self.somersaults = somersaults
        self.twists = twists

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
            self.tau_hips_min, self.tau_hips_init, self.tau_hips_max = -300, 0, 300  # hips and torso
            self.high_torque_idx = [
                6 - self.nb_root,
                7 - self.nb_root,
                8 - self.nb_root,
                13 - self.nb_root,
                14 - self.nb_root,
            ]
            self.qddot_min, self.qddot_init, self.qddot_max = -1000, 0, 1000

            self.velocity_max = 100  # qdot
            self.velocity_max_phase_transition = 10  # qdot hips, thorax in phase 2

            self.random_scale = 0.02  # relative to the maximal bounds of the states or controls

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

            self._set_boundary_conditions()
            self._set_initial_guesses()
            self._set_initial_momentum()
            self._set_dynamics()
            # self._set_constraints()
            self._set_objective_functions()

            self._set_mapping()

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                self.n_shooting,
                self.phase_durations,
                x_init=self.x_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                # constraints=self.constraints,
                phase_transitions=self.phase_transitions,
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

        # --- Objective function --- #
        for i in range(len(self.n_shooting)):
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                derivative=True,
                key="qdot",
                index=(6, 7, 8, 9, 10, 11, 12, 13, 14),
                weight=20,
                phase=i,
            )  # Regularization
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=1,
                marker_index=6,
                weight=50,
                phase=i,
            )  # Right hand trajectory
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=1,
                marker_index=11,
                weight=50,
                phase=i,
            )  # Left hand trajectory
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=0,
                marker_index=16,
                weight=50,
                phase=i,
            )  # feet trajectory
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=(6, 7, 8, 13, 14), key="q", weight=10, phase=i
            )  # core DoFs
        # Minimize initial angular momentum
        # self.objective_functions.add(
        #     ObjectiveFcn.Mayer.MINIMIZE_ANGULAR_MOMENTUM, phase=0, node=Node.START, weight=10000
        # )

        # Track momentum and Minimize delta momentum
        for i in range(2):
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_ANGULAR_MOMENTUM, phase=i,
                target=np.repeat(self.sigma0[:, np.newaxis], self.n_shooting[i], axis=1), weight=100
            )
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_LINEAR_MOMENTUM, index=[0, 1], phase=i,
                target=np.repeat(self.p0[:2, np.newaxis], self.n_shooting[i], axis=1), weight=100
            )
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_ANGULAR_MOMENTUM, phase=i, derivative=True, weight=100
            )
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_LINEAR_MOMENTUM, index=[0, 1], phase=i, derivative=True, weight=100
            )

        # Help to stay upright at the landing.
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            index=(0, 1, 2),
            target=[0, 0, 0],
            key="q",
            weight=0.1,
            phase=1,
            node=Node.END,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            index=3,
            target=self.somersaults - self.thorax_hips_xyz - self.slack_final_somersault / 2,
            key="q",
            weight=0.1,
            phase=1,
            node=Node.END,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE, index=4, target=0, key="q", weight=0.1, phase=1, node=Node.END
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE, index=5, target=self.twists, key="q", weight=0.1, phase=1, node=Node.END
        )

        slack_duration = 0.15
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=self.phase_durations[0] - slack_duration,
            max_bound=self.phase_durations[0] + slack_duration,
            phase=0,
            weight=1e-6,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=self.phase_durations[1] - slack_duration,
            max_bound=self.phase_durations[1] + slack_duration,
            phase=1,
            weight=1e-6,
        )


    def _set_constraints(self):
        # --- Constraints --- #
        # Set time as a variable
        slack_duration = 0.3
        # self.constraints.add(ConstraintFcn.TIME_CONSTRAINT,
        #                      node=Node.END,
        #                      min_bound=7/8 * self.duration - 1/2*slack_duration,
        #                      max_bound=7/8 * self.duration + 1/2*slack_duration, phase=0)
        # self.constraints.add(ConstraintFcn.TIME_CONSTRAINT,
        #                      node=Node.END,
        #                      min_bound=1/8 * self.duration - 1/2*slack_duration,
        #                      max_bound=1/8 * self.duration + 1/2*slack_duration, phase=1)
    def _set_initial_momentum(self):
        q_init = self.x_bounds[0].min[:self.n_q, 0]
        qdot_init = self.x_bounds[0].min[self.n_q:, 0]

        m = brd.Model(self.biorbd_model_path)
        self.sigma0 = m.angularMomentum(q_init, qdot_init).to_array()
        self.p0 = m.mass() * m.CoMdot(q_init, qdot_init).to_array()



    def _set_initial_guesses(self):
        # --- Initial guess --- #
        total_n_shooting = np.sum(self.n_shooting) + len(self.n_shooting)
        # Initialize state vector
        # if self.x is None:
        self.x = np.zeros((self.n_q + self.n_qdot, total_n_shooting))

        # determine v such that final z == 0
        v0 = 1 / 2 * 9.81 * self.duration  #
        # time vector
        data_point = np.linspace(0, self.duration, total_n_shooting)
        # parabolic trajectory on Z
        self.x[2, :] = v0 * data_point + -9.81 / 2 * data_point ** 2
        # Somersaults
        self.x[3, :] = np.hstack(
            (
                np.linspace(0, self.phase_proportions[0] * self.somersaults, self.n_shooting[0] + 1),
                np.linspace(self.phase_proportions[0] * self.somersaults, self.somersaults, self.n_shooting[1] + 1),
            )
        )
        # Twists
        self.x[5, :] = np.hstack(
            (np.linspace(0, self.twists, self.n_shooting[0] + 1), self.twists * np.ones(self.n_shooting[1] + 1))
        )

        # Handle second DoF of arms with Noise.
        self.x[6:9, :] = np.random.random((3, total_n_shooting)) * np.pi / 12 - np.pi / 24
        self.x[10, :] = np.random.random((1, total_n_shooting)) * np.pi / 2 - (np.pi - np.pi / 4)
        self.x[12, :] = np.random.random((1, total_n_shooting)) * np.pi / 2 + np.pi / 4
        self.x[13:15, :] = np.random.random((2, total_n_shooting)) * np.pi / 12 - np.pi / 24

        # velocity on Y
        self.x[self.n_q + 0, :] = self.velocity_x
        self.x[self.n_q + 1, :] = self.velocity_y
        self.x[self.n_q + 2, :] = self.vertical_velocity_0 - 9.81 * data_point
        # Somersaults rate
        self.x[self.n_q + 3, :] = self.somersault_rate_0
        # Twists rate
        self.x[self.n_q + 5, :] = self.twists / self.duration

        # random for other velocities
        self.x[self.n_q + 6:, :] = (
                (np.random.random((self.n_qdot - self.nb_root, total_n_shooting)) * 2 - 1)
                * self.velocity_max
                * self.random_scale
        )

        # random for other velocities in phase 2 to only
        low_speed_idx = [self.n_q + 6, self.n_q + 7, self.n_q + 8, self.n_q + 13, self.n_q + 14]
        n_shooting_phase_0 = self.n_shooting[0] + 1
        n_shooting_phase_1 = self.n_shooting[1] + 1
        self.x[low_speed_idx, n_shooting_phase_0:] = (
                (np.random.random((len(low_speed_idx), n_shooting_phase_1)) * 2 - 1)
                * self.velocity_max_phase_transition
                * self.random_scale
        )

        self._set_initial_states(self.x)
        self._set_initial_controls()

    def _set_initial_states(self, X0: np.array = None):
        if X0 is None:
            self.x_init.add([0] * (self.n_q + self.n_q))
        else:
            mesh_point_init = 0
            for i in range(self.n_phases):
                self.x_init.add(
                    X0[:, mesh_point_init: mesh_point_init + self.n_shooting[i] + 1],
                    interpolation=InterpolationType.EACH_FRAME,
                )
                mesh_point_init += self.n_shooting[i]

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None and self.u is None:
            for phase in range(len(self.n_shooting)):
                n_shooting = self.n_shooting[phase]
                tau_J_random = np.random.random((self.n_tau, n_shooting)) * 2 - 1

                tau_max = self.tau_max * np.ones(self.n_tau)
                tau_max[self.high_torque_idx] = self.tau_hips_max
                tau_J_random = tau_J_random * tau_max[:, np.newaxis] * self.random_scale

                qddot_J_random = (
                        (np.random.random((self.n_tau, n_shooting)) * 2 - 1) * self.qddot_max * self.random_scale
                )
                qddot_B_random = (
                        (np.random.random((self.nb_root, n_shooting)) * 2 - 1) * self.qddot_max * self.random_scale
                )

                if self.dynamics_type == "explicit":
                    self.u_init.add(tau_J_random, interpolation=InterpolationType.EACH_FRAME)
                elif self.dynamics_type == "root_explicit":
                    self.u_init.add(qddot_J_random, interpolation=InterpolationType.EACH_FRAME)
                elif self.dynamics_type == "implicit":
                    u = np.vstack((tau_J_random, qddot_B_random, qddot_J_random))
                    self.u_init.add(u, interpolation=InterpolationType.EACH_FRAME)
                elif self.dynamics_type == "root_implicit":
                    u = np.vstack((qddot_B_random, qddot_J_random))
                    self.u_init.add(u, interpolation=InterpolationType.EACH_FRAME)
                else:
                    raise ValueError("Check spelling, choices are explicit, root_explicit, implicit, root_implicit")
        elif self.u is not None:
            for phase in range(len(self.n_shooting)):
                self.u_init.add(self.u[phase][:, :-1], interpolation=InterpolationType.EACH_FRAME)
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)

                shooting = 0
                for i in range(len(self.n_shooting)):
                    self.u_init.add(
                        U0[:, shooting: shooting + self.n_shooting[i]], interpolation=InterpolationType.EACH_FRAME
                    )
                    shooting += self.n_shooting[i]

    def _set_boundary_conditions(self):
        self.x_bounds = BoundsList()

        tilt_bound = np.pi / 4
        tilt_final_bound = np.pi / 12  # 15 degrees

        initial_arm_elevation = 2.8
        arm_rotation_z_upp = np.pi / 2
        arm_rotation_z_low = 1
        arm_elevation_y_low = 0.01
        arm_elevation_y_upp = np.pi - 0.01
        thorax_hips_xyz = np.pi / 6
        self.thorax_hips_xyz = thorax_hips_xyz
        arm_rotation_y_final = 2.4

        # slack_initial_vertical_velocity = 2
        # slack_initial_somersault_rate = 3
        # slack_initial_translation_velocities = 1
        slack_initial_translation_velocities = 0
        slack_initial_vertical_velocity = 0
        slack_initial_somersault_rate = 0

        # end phase 0
        slack_somersault = 30 * 3.14 / 180
        slack_twist = 30 * 3.14 / 180

        slack_final_somersault = np.pi / 24  # 7.5 degrees
        self.slack_final_somersault = slack_final_somersault
        slack_final_twist = np.pi / 24  # 7.5 degrees
        slack_final_dofs = np.pi / 24  # 7.5 degrees

        x_min = np.zeros((2, self.n_q + self.n_qdot, 3))
        x_max = np.zeros((2, self.n_q + self.n_qdot, 3))

        x_min[0, : self.n_q, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -initial_arm_elevation, 0, initial_arm_elevation, 0, 0]
        x_min[0, self.n_q:, 0] = [
            self.velocity_x - slack_initial_translation_velocities,
            self.velocity_y - slack_initial_translation_velocities,
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
        x_max[0, self.n_q:, 0] = [
            self.velocity_x - slack_initial_translation_velocities,
            self.velocity_y - slack_initial_translation_velocities,
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
        x_min[0, self.n_q:, 1] = -self.velocity_max

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
        x_max[0, self.n_q:, 1] = +self.velocity_max

        x_min[0, : self.n_q, 2] = [
            -3,
            -3,
            -0.001,
            self.phase_proportions[0] * self.somersaults - slack_somersault,
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
        x_min[0, self.n_q:, 2] = -self.velocity_max

        x_max[0, : self.n_q, 2] = [
            3,
            3,
            10,
            self.phase_proportions[0] * self.somersaults + slack_somersault,
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
        x_max[0, self.n_q:, 2] = +self.velocity_max

        x_min[1, : self.n_q, 0] = x_min[0, : self.n_q, 2]
        x_min[1, self.n_q:, 0] = x_min[0, self.n_q:, 2]

        x_max[1, : self.n_q, 0] = x_max[0, : self.n_q, 2]
        x_max[1, self.n_q:, 0] = x_max[0, self.n_q:, 2]

        x_min[1, : self.n_q, 1] = [
            -3,
            -3,
            -0.001,
            self.phase_proportions[0] * self.somersaults - slack_final_somersault,
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
        x_min[1, self.n_q:, 1] = [
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
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
        ]  # x_max[0, :self.n_q, 1]
        x_max[1, self.n_q:, 1] = [
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
        ]

        x_min[1, : self.n_q, 2] = [
            -0.15,
            -0.25,
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
        x_min[1, self.n_q:, 2] = [
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
        ]

        x_max[1, : self.n_q, 2] = [
            0.15,
            0.25,
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
        x_max[1, self.n_q:, 2] = [
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
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
                self.u_bounds[0].min[self.high_torque_idx, :] = self.tau_hips_min
                self.u_bounds[0].max[self.high_torque_idx, :] = self.tau_hips_max
            elif self.dynamics_type == "root_explicit":
                self.u_bounds.add([self.qddot_min] * self.n_qddot, [self.qddot_max] * self.n_qddot)
            elif self.dynamics_type == "implicit":
                self.u_bounds.add(
                    [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
                    [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
                )
                self.u_bounds[0].min[self.high_torque_idx, :] = self.tau_hips_min
                self.u_bounds[0].max[self.high_torque_idx, :] = self.tau_hips_max
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

    def _load_initial_guess(self):

        file_name = (
            "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/other/root_explicit_2022-02-09_18_45_25.bo"
        )
        ocp, data = OptimalControlProgram.load(file_name)
        q = np.hstack((data.states[0]["q"], data.states[1]["q"]))
        qdot = np.hstack((data.states[0]["qdot"], data.states[1]["qdot"]))
        states = np.hstack((data.states[0]["all"], data.states[1]["all"]))
        self.x = states
        self.p = np.array(data.parameters["time"])
        print(self.p)
