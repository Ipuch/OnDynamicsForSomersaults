"""
This script is reading and organizing the raw data results from Miller Optimal control problems into a nice DataFrame.
It requires the all the raw data to run the script.
"""
import os
from pathlib import Path
import pickle
from bioptim import OptimalControlProgram, Shooting
import biorbd
from custom_dynamics.enums import MillerDynamics
from utils import (
    stack_states,
    stack_controls,
    define_integrated_time,
    define_time,
    angular_momentum_time_series,
    linear_momentum_time_series,
    comdot_time_series,
    comddot_time_series,
    angular_momentum_deviation,
    linear_momentum_deviation,
    define_control_integrated,
    residual_torque_time_series,
    root_explicit_dynamics,
)
import numpy as np
import pandas as pd

out_path_raw = "../../OnDynamicsForSommersaults_results/raw_last01-03-22"
model = "../Model_JeCh_15DoFs.bioMod"
# open files
files = os.listdir(out_path_raw)
files.sort()

column_names = [
    "model_path",
    "irand",
    "extra_obj",
    "computation_time",
    "cost",
    "detailed_cost",
    "iterations",
    "status",
    "states" "controls",
    "parameters",
    "dynamics_type",
    "q_integrated",
    "qdot_integrated",
    "qddot_integrated",
    "n_shooting",
    "n_theads",
]
df_results = pd.DataFrame(columns=column_names)

for i, file in enumerate(files):
    if file.endswith(".pckl"):
        print(file)
        p = Path(f"{out_path_raw}/{file}")
        file_path = open(p, "rb")
        data = pickle.load(file_path)

        # DM to array
        data["cost"] = np.array(data["cost"])[0][0]
        data["parameters"]["time"] = np.array(data["parameters"]["time"]).T[0]

        # fill qddot_integrated
        if (
            data["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
            or data["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT
        ):
            p = p.with_suffix(".bo")
            ocp, sol = OptimalControlProgram.load(p.resolve().__str__())
            sol_integrated = sol.integrate(
                shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, merge_phases=True, continuous=False
            )
            data["qddot_integrated"] = sol_integrated.states["qddot"]
        else:
            data["qddot_integrated"] = np.nan

        df_dictionary = pd.DataFrame([data])
        df_results = pd.concat([df_results, df_dictionary], ignore_index=True)

# df_results.to_pickle("Dataframe_results_5.pkl")
# df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")

# fill new columns
n_row = len(df_results)
df_results["t"] = None
df_results["t_integrated"] = None
df_results["tau_integrated"] = None
df_results["q"] = None
df_results["qdot"] = None
df_results["qddot"] = None
df_results["tau"] = None
df_results["int_T"] = None
df_results["int_R"] = None
df_results["angular_momentum"] = None
df_results["linear_momentum"] = None
df_results["comdot"] = None
df_results["comddot"] = None
df_results["angular_momentum_rmse"] = None
df_results["linear_momentum_rmse"] = None
df_results["T1"] = None
df_results["T2"] = None

m = biorbd.Model(model)
n_step = 5
N = 2
N_integrated = 2
for index, row in df_results.iterrows():
    print(index)
    t_integrated = define_integrated_time(row.parameters["time"], row.n_shooting, n_step)
    q_integrated = row.q_integrated
    qdot_integrated = row.qdot_integrated
    N_integrated = len(t_integrated)

    if row.dynamics_type == MillerDynamics.IMPLICIT or row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
        qddot_integrated = define_control_integrated(row.controls, n_step, "qddot")

    elif row.dynamics_type == MillerDynamics.EXPLICIT:
        tau_integrated = define_control_integrated(row.controls, n_step, "tau")
        tau_integrated = np.vstack((np.zeros((6, N_integrated)), tau_integrated))
        qddot_integrated = np.zeros((m.nbQ(), N_integrated))
        # apply forward dynamics
        for ii in range(N_integrated):
            qddot_integrated[:, ii] = m.ForwardDynamics(
                q_integrated[:, ii], qdot_integrated[:, ii], tau_integrated[:, ii]
            ).to_array()
    elif (
        row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
        or row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT
    ):
        qddot_integrated = row.qddot_integrated

    elif row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
        qddot_joints_integrated = define_control_integrated(row.controls, n_step, key="qddot_joint")
        qddot_integrated = np.zeros((m.nbQ(), N_integrated))
        qddot_integrated[6:, :] = qddot_joints_integrated
        # apply forward dynamics
        for i in range(N_integrated):
            qddot_integrated[:6, i] = root_explicit_dynamics(
                m, q_integrated[:, i], qdot_integrated[:, i], qddot_joints_integrated[:, i]
            )

    # compute metric related to integrated variables.
    # Translation residuals
    T = residual_torque_time_series(m, q_integrated, qdot_integrated, qddot_integrated)[:3]
    # compute the norm and the integral of it
    T = np.linalg.norm(T, axis=0)
    int_T = np.zeros(1)
    for j in range(T.shape[0] - 1):
        dt = np.diff(t_integrated[j : j + 2])[0]
        if dt != 0:
            int_T += np.trapz(T[j : j + 2], dx=dt)

    # Rotation residuals
    R = residual_torque_time_series(m, q_integrated, qdot_integrated, qddot_integrated)[3:]
    R = np.linalg.norm(R, axis=0)
    int_R = np.zeros(1)
    for j in range(R.shape[0] - 1):
        dt = np.diff(t_integrated[j : j + 2])[0]
        if dt != 0:
            int_R += np.trapz(R[j : j + 2], dx=dt)

    # store also all tau_integrated (already computed for EXPLICIT)
    if row.dynamics_type != MillerDynamics.EXPLICIT:
        if (
            row.dynamics_type == MillerDynamics.IMPLICIT
            or row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
        ):
            # tau_integrated = define_control_integrated(row.controls, n_step, "tau")
            # tau_integrated = np.vstack((np.zeros((6, N_integrated)), tau_integrated))
            tau_integrated = np.zeros((m.nbQ(), N_integrated))
            for ii in range(N_integrated):
                tau_integrated[:, ii] = m.InverseDynamics(
                    q_integrated[:, ii], qdot_integrated[:, ii], qddot_integrated[:, ii]
                ).to_array()
        else:
            tau_integrated = np.zeros((m.nbQ(), N_integrated))
            for ii in range(N_integrated):
                tau_integrated[:, ii] = m.InverseDynamics(
                    q_integrated[:, ii], qdot_integrated[:, ii], qddot_integrated[:, ii]
                ).to_array()

    # non integrated values, at nodes.
    t = define_time(row.parameters["time"], row.n_shooting)
    N = len(t)
    q = stack_states(row.states, "q")
    qdot = stack_states(row.states, "qdot")

    # compute tau only for non root dynamics because qddot is not needed
    if row.dynamics_type == MillerDynamics.IMPLICIT:
        tau = stack_controls(row.controls, "tau")

    elif row.dynamics_type == MillerDynamics.EXPLICIT:
        tau = stack_controls(row.controls, "tau")

    elif row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
        tau = stack_controls(row.controls, "tau")

    # compute qddot
    if row.dynamics_type == MillerDynamics.IMPLICIT or row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
        qddot = stack_controls(row.controls, "qddot")

    elif row.dynamics_type == MillerDynamics.EXPLICIT:
        qddot = np.zeros((m.nbQ(), N))
        for ii in range(N):
            qddot[:, ii] = m.ForwardDynamics(q[:, ii], qdot[:, ii], tau[:, ii]).to_array()

    elif row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
        qddot_joints = stack_controls(row.controls, key="qddot_joint")
        qddot = np.zeros((m.nbQ(), N))
        qddot[6:, :] = qddot_joints
        for i in range(N):
            qddot[:6, i] = root_explicit_dynamics(m, q[:, i], qdot[:, i], qddot_joints[:, i])

    elif (
        row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
        or row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT
    ):
        qddot = stack_states(row.states, "qddot")

    # compute tau for root dynamics because qddot is needed first
    if row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
        tau = np.zeros((m.nbQ(), N))
        for ii in range(N):
            tau[:, ii] = m.InverseDynamics(q[:, ii], qdot[:, ii], qddot[:, ii]).to_array()

    elif row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
        tau = np.zeros((m.nbQ(), N))
        for ii in range(N):
            tau[:, ii] = m.InverseDynamics(q[:, ii], qdot[:, ii], qddot[:, ii]).to_array()

    elif row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        tau = np.zeros((m.nbQ(), N))
        for ii in range(N):
            tau[:, ii] = m.InverseDynamics(q[:, ii], qdot[:, ii], qddot[:, ii]).to_array()

    # non integrated metrics
    angular_momentum = angular_momentum_time_series(m, q, qdot)
    linear_momentum = linear_momentum_time_series(m, q, qdot)
    comdot = comdot_time_series(m, q, qdot)
    comddot = comddot_time_series(m, q, qdot, qddot)
    mass = m.mass()
    angular_momentum_rmse = angular_momentum_deviation(angular_momentum)
    linear_momentum_rmse = linear_momentum_deviation(mass, comdot, t, comddot)

    df_results.at[index, "t"] = t
    df_results.at[index, "t_integrated"] = t_integrated
    df_results.at[index, "qddot_integrated"] = qddot_integrated
    df_results.at[index, "tau_integrated"] = tau_integrated
    df_results.at[index, "q"] = q
    df_results.at[index, "qdot"] = qdot
    df_results.at[index, "qddot"] = qddot
    df_results.at[index, "tau"] = tau
    df_results.at[index, "int_T"] = int_T[0]
    df_results.at[index, "int_R"] = int_R[0]
    df_results.at[index, "angular_momentum"] = angular_momentum
    df_results.at[index, "linear_momentum"] = linear_momentum
    df_results.at[index, "comdot"] = comdot
    df_results.at[index, "comddot"] = comddot
    df_results.at[index, "angular_momentum_rmse"] = angular_momentum_rmse
    df_results.at[index, "linear_momentum_rmse"] = linear_momentum_rmse
    df_results.at[index, "T1"] = row.parameters["time"][0]
    df_results.at[index, "T2"] = row.parameters["time"][1]

# EXTRA COMPUTATIONS
# NICE LATEX LABELS
df_results["dynamics_type_label"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "dynamics_type_label"] = r"$\text{Full-Exp}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dynamics_type_label"
] = r"$\text{Base-Exp}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dynamics_type_label"
] = r"$\text{Full-Imp-}\ddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dynamics_type_label"
] = r"$\text{Base-Imp-}\ddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dynamics_type_label"
] = r"$\text{Full-Imp-}\dddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dynamics_type_label"
] = r"$\text{Base-Imp-}\dddot{q}$"

# COST FUNCTIONS
# four first functions for each phase
df_results["cost_J"] = None
df_results["cost_angular_momentum"] = None

for ii in range(10):
    df_results[f"cost_J{ii}"] = None

for index, row in df_results.iterrows():
    print(index)
    dc = row.detailed_cost

    # Index of cost functions in details costs
    idx_angular_momentum = [5, 6]

    if (
        row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT
        or row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
        or row.dynamics_type == MillerDynamics.IMPLICIT
        or row.dynamics_type == MillerDynamics.ROOT_IMPLICIT
    ):
        idx_J = [
            0,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            1,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            2,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            3,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker foot
            4,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            11,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            12,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            13,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            14,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            15,
        ]  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
    else:
        idx_J = [
            0,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            1,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            2,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            3,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker foot
            4,  # Phase 1 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            10,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, derivative=True, key = qdot
            11,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            12,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, derivative=True, marker hand
            13,  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof
            14,
        ]  # Phase 2 ObjectiveFcn.Lagrange.MINIMIZE_STATE, key = q # core dof

    for ii, idx in enumerate(idx_J):
        df_results.at[index, f"cost_J{ii}"] = row.detailed_cost[idx]["cost_value_weighted"]

    df_results.at[index, "cost_J"] = np.sum([row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_J])
    print(np.sum([row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_J]))

    df_results.at[index, "cost_angular_momentum"] = np.sum(
        [row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_angular_momentum]
    )


df_results["grps"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "grps"] = "Explicit"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "grps"] = "Root_Explicit"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "grps"] = "Implicit_qddot"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "grps"] = "Root_Implicit_qddot"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "grps"] = "Implicit_qdddot"
df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "grps"] = "Root_Implicit_qdddot"

# df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")
# df_results.to_pickle("Dataframe_results_metrics_5.pkl")

# COMPUTE CLUSTERS of same value for Cost_J
df_results["main_cluster"] = False
# specify the value for each dynamic type
cost_J_cluster_values = [10.63299, 10.61718, 2.68905, 2.591474, 10.58839, 10.58604]
for index, row in df_results.iterrows():
    if row.dynamics_type == MillerDynamics.EXPLICIT:
        cluster_val = cost_J_cluster_values[0]
    elif row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
        cluster_val = cost_J_cluster_values[1]
    elif row.dynamics_type == MillerDynamics.IMPLICIT:
        cluster_val = cost_J_cluster_values[2]
    elif row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
        cluster_val = cost_J_cluster_values[3]
    elif row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
        cluster_val = cost_J_cluster_values[4]
    elif row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        cluster_val = cost_J_cluster_values[5]

    if abs(cluster_val - row["cost_J"]) < 1e-3:
        print(row.dynamics_type)
        print(cluster_val - row["cost_J"])
        print(row.irand)
        df_results.at[index, "main_cluster"] = True

# saves the dataframe
df_results.to_pickle("Dataframe_results_metrics_5.pkl")
