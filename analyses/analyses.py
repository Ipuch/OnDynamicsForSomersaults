import numpy as np
import matplotlib.pyplot as plt
from bioptim import Shooting
from scipy.stats import linregress


def graphs_analyses(pkl_name):
    return


def Analyses(sol, dynamics_type, out_path_secondary_variables):

    q = sol.states["q"]
    qdot = sol.states["qdot"]
    N = np.shape(q)[1]
    if dynamics_type == "explicit" or dynamics_type == "implicit":
        tau = sol.controls["tau"]
    elif dynamics_type == "root_explicit" or dynamics_type == "root_implicit":
        qddot = sol.comtrols["qddot"]

    sol_integrated = sol.integrate(
        shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, merge_phases=True, continuous=False
    )

    q_integrated = sol_integrated.states["q"]
    qdot_integrated = sol_integrated.states["qdot"]
    N_integrated = np.shape(q_integrated)[1]
    if dynamics_type == "explicit" or dynamics_type == "implicit":
        tau_integrated = sol_integrated.controls["tau"]
    elif dynamics_type == "root_explicit" or dynamics_type == "root_implicit":
        qddot_integrated = sol_integrated.comtrols["qddot"]

    m = sol.nlp.biorbd_model

    angular_momentum = np.zeros((3, N))
    angular_momentum_norm = np.zeros((N,))
    linear_momentum = np.zeros((3, N))
    for node in range(N):
        angular_momentum[:, node] = m.angularMomentum(q, qdot, True).to_array()
        angular_momentum_norm[node] = np.linalg.norm(angular_momentum[:, node]).to_array()
        linear_momentum[:, node] = m.CoMdot(q, qdot, True).to_array() * m.mass().to_array()

    angular_momentum_norm_mean = np.mean(angular_momentum_norm)
    angular_momentum_rmsd = np.sqrt(((angular_momentum_norm - angular_momentum_norm_mean) ** 2).mean())

    slope, intercept, p, linear_momentum_rmsd, intercept_std = linregress(x, y)

    if dynamics_type == "root_explicit" or dynamics_type == "root_implicit":
        residual_tau_integrated = np.zeros((m.nbRoot(), N_integrated))
        for node_integrated in range(N_integrated):
            residual_tau_integrated = m.inverseDynamics(q_integrated, qdot_integrated, qddot_integrated).to_array()[:6]
    else:
        residual_tau_integrated = 0

    residual_tau_sum = np.sum(np.abs(residual_tau))

    f = open(f"{out_path_secondary_variables}/miller_{dynamics_type}_irand{i_rand}_analyses.pckl", "wb")
    data = {
        "angular_momentum": angular_momentum,
        "angular_momentum_norm": angular_momentum_norm,
        "linear_momentum": linear_momentum,
        "residual_tau_integrated ": residual_tau_integrated,
        "angular_momentum_rmsd": angular_momentum_rmsd,
        "linear_momentum_rmsd": linear_momentum_rmsd,
        "residual_tau_sum": residual_tau_sum,
    }

    pickle.dump(data, f)
    f.close()

    return
