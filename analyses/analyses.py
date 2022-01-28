
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import biorbd
import os
import pickle


def graphs_analyses(angular_momentum_rmsd_all, linear_momentum_rmsd_all, residual_tau_sum_all, computation_time_all, cost_all, iterations_all):

    colors = ['#2E5A90FF', '#00BA87FF', '#DDEA00FF', '#BE2AD0FF', '#76DF1FFF', '#13BBF2FF', '#500375FF']
    shift = [-0.3, -0.1, 0.1, 0.3]
    labels = ["Angular Momentum RMSD",
              "Linear Momentum RMSD",
              "Resirual Tau",
              "Computation Time",
              "Optimal Cost",
              "Number of iterations"]

    angular_momentum_rmsd_all_mean = np.mean(angular_momentum_rmsd_all, axis=1)
    linear_momentum_rmsd_all_mean = np.mean(linear_momentum_rmsd_all, axis=1)
    residual_tau_sum_all_mean = np.mean(residual_tau_sum_all, axis=1)
    computation_time_all_mean = np.mean(computation_time_all, axis=1)
    cost_all_mean = np.mean(cost_all, axis=1)
    iterations_all_mean = np.mean(iterations_all, axis=1)

    angular_momentum_rmsd_all_std = np.std(angular_momentum_rmsd_all, axis=1)
    linear_momentum_rmsd_all_std = np.std(linear_momentum_rmsd_all, axis=1)
    residual_tau_sum_all_std = np.std(residual_tau_sum_all, axis=1)
    computation_time_all_std = np.std(computation_time_all, axis=1)
    cost_all_std = np.std(ancost_all, axis=1)
    iterations_all_std = np.std(iterations_all, axis=1)

    variables_list = [angular_momentum_rmsd_all, linear_momentum_rmsd_all, residual_tau_sum_all, computation_time_all,
                      cost_all, iterations_all]
    variables_mean_list = [angular_momentum_rmsd_all_mean, linear_momentum_rmsd_all_mean, residual_tau_sum_all_mean,
                           computation_time_all_mean, cost_all_mean, iterations_all_mean]
    variables_std_list = [angular_momentum_rmsd_all_std, linear_momentum_rmsd_all_std, residual_tau_sum_all_std,
                           computation_time_all_std, cost_all_std, iterations_all_std]

    fig, ax = plt.subplots(1, 1, tight_layout=True)

    for j in range(6):
        for i in range(4):
            ax.bar(j + shift[i],variables_mean_list[j][i] + variables_std_list[j][i],
                            color=colors[i],
                            bottom=variables_mean_list[j][i] - variables_std_list[j][i],
                            label=labels[j],
                            alpha=0.3)

            plt.plot(j + shift[i], variables_list[j][i, :], '.', color=colors[i])

    plt.legend()
    plt.show()
    plt.savefig('Comparaison.png', dpi=900)

    return

def Analyses(out_path_raw, file_name, out_path_secondary_variables):

    file = open(f"{out_path_raw}/{file_name}", "rb")
    data = pickle.load(file)

    q = np.hstack((data["states"][0]['q'], data["states"][1]['q']))
    qdot = np.hstack((data["states"][0]['qdot'], data["states"][1]['qdot']))
    t = data["parameters"]['time']
    time = np.hstack((np.linspace(0, float(t[0]), np.shape(data["states"][0]['q'])[1]), np.linspace(float(t[0]), float(t[0])+float(t[1]), np.shape(data["states"][1]['q'])[1]))) # [1:]
    N = np.shape(q)[1]
    q_integrated = data["q_integrated"]
    qdot_integrated = data["qdot_integrated"]
    N_integrated = np.shape(q_integrated)[1]
    dynamics_type = data["dynamics_type"]
    computation_time = data["computation_time"]
    cost = data["cost"]
    iterations = data["iterations"]

    if dynamics_type == "explicit" or dynamics_type == "implicit":
        tau = np.hstack((data["controls"][0]['tau'], data["controls"][1]['tau']))
        tau_integrated = np.hstack((np.repeat(data["controls"][0]['tau'], 5, axis=1)[:, :-4],
                                    np.repeat(data["controls"][1]['tau'], 5, axis=1)[:, :-4]))
    elif dynamics_type == "root_explicit" or dynamics_type == "root_implicit":
        qddot = np.hstack((data["controls"][0]['qddot'], data["controls"][1]['qddot']))
        qddot_integrated = np.hstack((np.repeat(data["controls"][0]['qddot'], 5, axis=1)[:, :-4],
                                    np.repeat(data["controls"][1]['qddot'], 5, axis=1)[:, :-4]))

    m = biorbd.Model("../" + data["model_path"])

    angular_momentum = np.zeros((3, N))
    angular_momentum_norm = np.zeros((N, ))
    linear_momentum = np.zeros((3, N))
    CoM_position = np.zeros((3, N))
    for node in range(N):
        angular_momentum[:, node] = m.angularMomentum(q[:, node], qdot[:, node], True).to_array()
        angular_momentum_norm[node] = np.linalg.norm(angular_momentum[:, node])
        linear_momentum[:, node] = m.CoMdot(q[:, node], qdot[:, node], True).to_array() * m.mass()
        CoM_position[:, node] = m.CoM(q[:, node], True).to_array()

    angular_momentum_mean = np.mean(angular_momentum, axis=1)
    angular_momentum_rmsd = np.zeros((3, ))
    linear_momentum_rmsd = np.zeros((3,))
    for i in range(3):
        angular_momentum_rmsd[i] = np.sqrt(((angular_momentum[i, :] - angular_momentum_mean[i]) ** 2).mean())
        slope, intercept, p, linear_momentum_rmsd[i], intercept_std = linregress(time, CoM_position[i, :])

    if dynamics_type == "root_explicit" or dynamics_type == "root_implicit":
        residual_tau_integrated = np.zeros((m.nbRoot(), N_integrated))
        for node_integrated in range(N_integrated):
            residual_tau_integrated = m.inverseDynamics(q_integrated, qdot_integrated, qddot_integrated).to_array()[:6]
    else:
        residual_tau_integrated = 0

    residual_tau_sum = np.sum(np.abs(residual_tau_integrated))

    f = open(f"{out_path_secondary_variables}/miller_{dynamics_type}_irand{i_rand}_analyses.pckl", "wb")
    data_secondary = {"angular_momentum" : angular_momentum,
            "angular_momentum_norm" : angular_momentum_norm,
            "linear_momentum" : linear_momentum,
            "CoM_position" : CoM_position,
            "residual_tau_integrated " : residual_tau_integrated,
            "angular_momentum_rmsd" : angular_momentum_rmsd,
            "linear_momentum_rmsd" : linear_momentum_rmsd,
            "residual_tau_sum" : residual_tau_sum}

    pickle.dump(data_secondary , f)
    f.close()

    return np.sum(np.abs(angular_momentum_rmsd)), np.sum(np.abs(linear_momentum_rmsd)), np.sum(residual_tau_sum), computation_time, cost, iterations




out_path_raw = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw"
out_path_secondary_variables = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/secondary_variables"

#explicit, #root_explicit, #implicit, #root_implicit
angular_momentum_rmsd_all = np.zeros((4, 100))
linear_momentum_rmsd_all = np.zeros((4, 100))
residual_tau_sum_all = np.zeros((4, 100))
computation_time_all = np.zeros((4, 100))
cost_all = np.zeros((4, 100))
iterations_all = np.zeros((4, 100))

for file in os.listdir(out_path_raw):
    if file[-5:] == ".pckl":
        if "explicit" in file:
            if "root_" in file:
                i = 1
            else:
                i = 0
        if "implicit" in file:
            if "root_" in file:
                i = 3
            else:
                i = 2
        idx_irand = file.name.find("irand") + 5
        idx_pckl = file.name.find(".pckl")
        j = int(file.name[idx_irand : idx_pckl])
        angular_momentum_rmsd_all[i, j], linear_momentum_rmsd[i, j], residual_tau_sum[i, j], computation_time[i, j], cost[i, j], iterationsAnalyses(out_path_raw, file, out_path_secondary_variables)



graphs_analyses(angular_momentum_rmsd_all, linear_momentum_rmsd_all, residual_tau_sum_all, computation_time_all, cost_all, iterations_all)

