import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import biorbd
import os
import pickle
import bioviz
import bioptim
import seaborn as sns

"""
Code permettant de loader les données des optimisations et de faire les graphiques suivants :
figure_type_3: variables secondaires (moment cinétique/linéaire, tau résiduels, cost, intération, ...)
figure_type_4: Q, Qdot, Qddot, U pour tous les essais + min cost en gras
figure_type_5: barplot de la composition du cost
"""

global colors, shift, dynamics_types, figure_type_3, figure_type_4, figure_type_5


def root_explicit_dynamics(m, q, qdot, qddot_joints):
    mass_matrix_nl_effects = m.InverseDynamics(q, qdot, np.hstack((np.zeros((6,)), qddot_joints))).to_array()[:6]
    mass_matrix = m.massMatrix(q).to_array()
    qddot_base = -np.linalg.solve(mass_matrix[:6, :6], np.eye(6)) @ mass_matrix_nl_effects
    return qddot_base


def graphs_analyses(variables_list):

    global colors, shift, dynamics_types, figure_type_3

    labels = [
        "Angular Momentum RMSD",
        "Linear Momentum RMSD",
        "Residual Tau",
        "Computation Time",
        "Optimal Cost",
        "Number of iterations",
    ]

    variables_mean_list = np.zeros((6, 6))
    variables_std_list = np.zeros((6, 6))
    for j, key in enumerate(variables_list.keys()):
        variables_mean_list[j, :] = np.nanmean(variables_list[key][:, :], axis=1)
        variables_std_list[j, :] = np.nanstd(variables_list[key][:, :], axis=1)

    if figure_type_3:
        for j, key in enumerate(variables_list.keys()):
            plt.figure()
            plt.xticks(shift, labels=dynamics_types)

            for i in range(4):
                plt.plot(
                    np.array([shift[i] - 0.05, shift[i] + 0.05]),
                    np.ones((2,)) * variables_mean_list[j, i],
                    color="k",
                    linewidth=2,
                    alpha=0.3,
                )

                if j == 0:
                    plt.bar(
                        shift[i],
                        2 * variables_std_list[j, i],
                        width=0.1,
                        color="k",
                        bottom=variables_mean_list[j, i] - variables_std_list[j, i],
                        label="mean $\pm$ std",
                        alpha=0.1,
                    )
                    plt.plot(
                        np.ones((100,)) * shift[i] + np.random.random(100) * 0.1 - 0.05,
                        variables_list[key][i, :],
                        ".",
                        color=colors[i],
                    )
                else:
                    plt.bar(
                        shift[i],
                        2 * variables_std_list[j, i],
                        width=0.1,
                        color="k",
                        bottom=variables_mean_list[j, i] - variables_std_list[j, i],
                        alpha=0.1,
                    )
                    plt.plot(
                        np.ones((100,)) * shift[i] + np.random.random(100) * 0.1 - 0.05,
                        variables_list[key][i, :],
                        ".",
                        color=colors[i],
                    )
            if labels[j] != "Optimal Cost":
                plt.yscale("log")
            plt.title(labels[j])
            plt.show()
            plt.savefig(f"Comparaison_separes_{labels[j]}.png", dpi=900)

    return


def Analyses(
    out_path_raw,
    file_name,
    out_path_secondary_variables,
    i_rand,
    i_dynamics_type,
    axs,
    axs_5,
    min_cost_varilables_values,
):
    """
    *note: [variable]_inetgated veut dire que ca vient de bioptim.[...].integrate
    """
    global colors, shift, figure_type_4, figure_type_5

    # ouvrir les fichiers de sortie de l'OCP en cours de lecture
    file = open(f"{out_path_raw}/{file_name}", "rb")
    data = pickle.load(file)

    status = data["status"]
    if status != 0:  # L'optimastion n'a pas convergé
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            min_cost_varilables_values,
        )

    else:
        m = biorbd.Model("../" + data["model_path"])

        # Variables de performance
        dynamics_type = data["dynamics_type"]
        computation_time = data["computation_time"]
        cost = data["cost"]
        iterations = data["iterations"]
        if figure_type_5:
            detailed_cost_dict = data["detailed_cost"]
            detailed_cost = np.zeros((len(detailed_cost_dict),))
            for i in range(len(detailed_cost_dict)):
                detailed_cost[i] = detailed_cost_dict[i]["cost_value_weighted"]

        # Reorganiser les variables parce que plus d'une phase
        ## ETATS
        q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))
        qdot = np.hstack((data["states"][0]["qdot"], data["states"][1]["qdot"]))
        q_integrated = data["q_integrated"]
        qdot_integrated = data["qdot_integrated"]

        ## NOEUDS
        N = np.shape(q)[1]
        N_integrated = (N - 2) * 6 + 2

        ## TEMPS
        t = data["parameters"]["time"]
        time = np.hstack(
            (
                np.linspace(0, float(t[0]), np.shape(data["states"][0]["q"])[1]),
                np.linspace(float(t[0]), float(t[0]) + float(t[1]), np.shape(data["states"][1]["q"])[1]),
            )
        )
        time_integrated = np.array([])
        for i in range(N - 1):
            if i != 125:
                time_integrated = np.hstack((time_integrated, np.linspace(time[i], time[i + 1], 6)))
            else:
                time_integrated = np.hstack((time_integrated, time[i]))

        ## CONTROLES (provenant de l'optimisation ou d'un calcul si pas dispo dans optimisation
        if dynamics_type == "explicit" or dynamics_type == "implicit":
            tau = np.hstack((data["controls"][0]["tau"], data["controls"][1]["tau"]))
            tau_integrated = np.hstack(
                (
                    np.repeat(data["controls"][0]["tau"], 6, axis=1)[:, :-5],
                    np.repeat(data["controls"][1]["tau"], 6, axis=1)[:, :-5],
                )
            )

        if dynamics_type == "root_implicit" or dynamics_type == "implicit":
            qddot = np.hstack((data["controls"][0]["qddot"], data["controls"][1]["qddot"]))
            qddot_integrated = np.hstack(
                (
                    np.repeat(data["controls"][0]["qddot"], 6, axis=1)[:, :-5],
                    np.repeat(data["controls"][1]["qddot"], 6, axis=1)[:, :-5],
                )
            )

        if dynamics_type == "root_explicit":
            qddot_joints = np.hstack((data["controls"][0]["qddot_joint"], data["controls"][1]["qddot_joint"]))
            qddot = np.zeros((m.nbQ(), N))
            qddot[6:, :] = qddot_joints
            qddot_joints_integrated = np.hstack(
                (
                    np.repeat(data["controls"][0]["qddot_joint"], 6, axis=1)[:, :-5],
                    np.repeat(data["controls"][1]["qddot_joint"], 6, axis=1)[:, :-5],
                )
            )
            qddot_integrated = np.zeros((m.nbQ(), N_integrated))
            qddot_integrated[6:, :] = qddot_joints_integrated

            for i in range(N):
                qddot[:6, i] = root_explicit_dynamics(m, q[:, i], qdot[:, i], qddot_joints[:, i])
            for i in range(N_integrated):
                qddot_integrated[:6, i] = root_explicit_dynamics(
                    m, q_integrated[:, i], qdot_integrated[:, i], qddot_joints_integrated[:, i]
                )

        if dynamics_type == "explicit":
            residual_tau_integrated = 0
            tau = np.vstack((np.zeros((6, N)), tau))
            tau_integrated = np.vstack((np.zeros((6, N_integrated)), tau_integrated))
            qddot = np.zeros((m.nbQ(), N))
            for node in range(N):
                qddot[:, node] = m.ForwardDynamics(q[:, node], qdot[:, node], tau[:, node]).to_array()
            qddot_integrated = np.zeros((m.nbQ(), N_integrated))
            for node in range(N_integrated):
                qddot_integrated[:, node] = m.ForwardDynamics(
                    q_integrated[:, node], qdot_integrated[:, node], tau_integrated[:, node]
                ).to_array()

        else:
            if dynamics_type == "root_explicit":
                residual_tau_integrated = 0
            else:
                residual_tau_integrated = np.zeros((m.nbRoot(), N_integrated))
                for node_integrated in range(N_integrated):
                    residual_tau_integrated[:, node_integrated] = m.InverseDynamics(
                        q_integrated[:, node_integrated],
                        qdot_integrated[:, node_integrated],
                        qddot_integrated[:, node_integrated],
                    ).to_array()[:6]

            tau = np.zeros((m.nbQ(), N))
            for node in range(N):
                tau[:, node] = m.InverseDynamics(q[:, node], qdot[:, node], qddot[:, node]).to_array()

        index_continuous = [
            x for i, x in enumerate(np.arange(len(time_integrated))) if i != 125 * 6 + 1
        ]  # pour enlever les noeuds qui se répetent au changement de phase
        # remplissage des variables secondaires
        angular_momentum = np.zeros((3, N_integrated))
        angular_momentum_norm = np.zeros((N_integrated,))
        linear_momentum = np.zeros((3, N_integrated))
        CoM_position = np.zeros((3, N_integrated))
        CoM_velocity = np.zeros((3, N_integrated))
        CoM_acceleration = np.zeros((3, N_integrated))
        for node_integrated in range(N_integrated):
            angular_momentum[:, node_integrated] = m.angularMomentum(
                q_integrated[:, node_integrated], qdot_integrated[:, node_integrated], True
            ).to_array()
            angular_momentum_norm[node_integrated] = np.linalg.norm(angular_momentum[:, node_integrated])
            linear_momentum[:, node_integrated] = (
                m.CoMdot(q_integrated[:, node_integrated], qdot_integrated[:, node_integrated], True).to_array()
                * m.mass()
            )
            CoM_position[:, node_integrated] = m.CoM(q_integrated[:, node_integrated], True).to_array()
            CoM_velocity[:, node_integrated] = m.CoMdot(
                q_integrated[:, node_integrated], qdot_integrated[:, node_integrated], True
            ).to_array()
            CoM_acceleration[:, node_integrated] = m.CoMddot(
                q_integrated[:, node_integrated],
                qdot_integrated[:, node_integrated],
                qddot_integrated[:, node_integrated],
                True,
            ).to_array()

        angular_momentum_rmsd = np.zeros((3,))
        linear_momentum_rmsd = np.zeros((3,))
        for i in range(3):
            angular_momentum_rmsd[i] = np.sqrt(
                ((angular_momentum[i, index_continuous] - angular_momentum[i, 0]) ** 2).mean()
            )
            if i == 0 or i == 1:
                linear_momentum_rmsd[i] = m.mass() * np.sqrt(
                    ((CoM_velocity[i, index_continuous] - CoM_velocity[i, 0]) ** 2).mean()
                )
            else:
                linear_momentum_rmsd[i] = m.mass() * np.sqrt(
                    (
                        (
                            CoM_velocity[i, index_continuous]
                            - (CoM_acceleration[i, 0] * time_integrated[index_continuous] + CoM_velocity[i, 0])
                        )
                        ** 2
                    ).mean()
                )

        residual_tau_rms = np.sqrt(np.nanmean(residual_tau_integrated**2))

        f = open(f"{out_path_secondary_variables}/miller_{dynamics_type}_irand{i_rand}_analyses.pckl", "wb")
        data_secondary = {
            "angular_momentum": angular_momentum,
            "angular_momentum_norm": angular_momentum_norm,
            "linear_momentum": linear_momentum,
            "CoM_position": CoM_position,
            "residual_tau_integrated ": residual_tau_integrated,
            "angular_momentum_rmsd": angular_momentum_rmsd,
            "linear_momentum_rmsd": linear_momentum_rmsd,
            "residual_tau_rms": residual_tau_rms,
        }

        pickle.dump(data_secondary, f)
        f.close()

        if figure_type_4:
            for i in range(15):
                axs[0][i].plot(time, q[i, :], "-", color=colors[i_dynamics_type])
                axs[1][i].plot(time, qdot[i, :], "-", color=colors[i_dynamics_type])

                # if i_dynamics_type == 1 and i < 9:
                #     axs[2][i + 6].plot(time, qddot[i, :], "-", color=colors[i_dynamics_type])
                # elif i_dynamics_type != 1:
                axs[2][i].plot(time, qddot[i, :], "-", color=colors[i_dynamics_type])

                axs[3][i].step(time, tau[i, :], "-", color=colors[i_dynamics_type])

        if figure_type_5:
            micro_shift = np.linspace(-0.08, 0.08, 100)
            sorted_costs = np.sort(detailed_cost)  # a changer avec ke bon ordre pour tout le monde
            sorted_costs_idx = np.argsort(detailed_cost)
            bottom_cost = 0
            for j in range(len(detailed_cost)):
                if sorted_costs[j] > 1e-6:
                    axs_5.bar(
                        shift[i_dynamics_type] + micro_shift[i_rand],
                        sorted_costs[j],
                        width=0.08,
                        color=colors[sorted_costs_idx[j]],
                        bottom=bottom_cost,
                    )
                    bottom_cost += sorted_costs[j]

        if cost < min_cost_varilables_values["min_cost"][i_dynamics_type]:
            # si le cost est plus petit, voici le nouveau plus petit cout
            min_cost_varilables_values["time_min"][i_dynamics_type] = time
            min_cost_varilables_values["q_min"][i_dynamics_type] = q
            min_cost_varilables_values["qdot_min"][i_dynamics_type] = qdot
            min_cost_varilables_values["qddot_min"][i_dynamics_type] = qddot
            min_cost_varilables_values["tau_min"][i_dynamics_type] = tau
            min_cost_varilables_values["min_cost"][i_dynamics_type] = cost

        return (
            np.sum(np.abs(angular_momentum_rmsd)),
            np.sum(np.abs(linear_momentum_rmsd)),
            np.sum(residual_tau_rms),
            computation_time,
            cost,
            iterations,
            min_cost_varilables_values,
        )


################ parametres a changer ##################################################################################
# out_path_raw = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw"
out_path_raw = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw_all"
# out_path_secondary_variables = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/secondary_variables"
out_path_secondary_variables = (
    "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/secondary_variables"
)
animation_min_cost = False
figure_type_3 = False  # Variables secondaires
figure_type_4 = False  # Techniques Q, Qdot, Qddot, Tau
figure_type_5 = True  # Cost details
########################################################################################################################

# parametres pour les graphiques
colors = sns.color_palette(palette="coolwarm", n_colors=30)
shift = [-0.3, -0.1, 0.1, 0.3]
dynamics_types = ["Explicit", "Root explicit", "Implicit", "Root Implicit"]
Dof_names = [
    "Root Translation X",
    "Root Translation Y",
    "Root Translation Z",
    "Root Rotation X",
    "Root Rotation X",
    "Root Rotation X",
    "Thorax Rotation X",
    "Thorax Rotation Y",
    "Thorax Rotation Z",
    "Right Arm Rotation Z",
    "Right Arm Rotation Y",
    "Left Arm Rotation Z",
    "Left Arm Rotation Y",
    "Hips Rotation X",
    "Hips Rotation Y",
]
label_objectives = [  # a changer pour l'ordre quand l'OCP sera fixé (je ne peux pas faire automatiquement pcq les noms ne sont pas self explanatory
    "Minimize qdot derivative (except root) phase=0",
    "Minimize right hand trajectory phase=0",
    "Minimize left hand trajectory phase=0",
    "Minimize feet trajectory phase=0",
    "Minimize core DoFs (hips + thorax) phase=0",
    "Minimize angular momentum",
    "Minimize time phase=0",
    "Minimize qdot derivative (except root) phase=1",
    "Minimize right hand trajectory phase=1",
    "Minimize left hand trajectory phase=1",
    "Minimize feet trajectory phase=1",
    "Minimize core DoFs (hips + thorax) phase=1",
    "Minimize time phase=1",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
]


# prefill the minimum optimal data
min_cost = np.ones((4,)) * 1e30
q_min = [[], [], [], []]
qdot_min = [[], [], [], []]
qddot_min = [[], [], [], []]
tau_min = [[], [], [], []]
time_min = [[], [], [], []]

min_cost_varilables_values = {
    "q_min": q_min,
    "qdot_min": qdot_min,
    "qddot_min": qddot_min,
    "tau_min": tau_min,
    "time_min": time_min,
    "min_cost": min_cost,
}

axs = []
axs_5 = 0
if figure_type_4:
    # initialisation des figures pour Q, Qdot, Qddot, Tau
    fig_1, axs_1 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Q
    axs_1 = axs_1.ravel()
    fig_2, axs_2 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Qdot
    axs_2 = axs_2.ravel()
    fig_3, axs_3 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Qddot
    axs_3 = axs_3.ravel()
    fig_4, axs_4 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Tau
    axs_4 = axs_4.ravel()

    axs = [axs_1, axs_2, axs_3, axs_4]
    for j, ax in enumerate(axs):
        for k in range(4):
            ax[0].plot(0, 0, color=colors[k], label=dynamics_types[k])
            ax[0].legend()
        for i in range(15):
            ax[i].set_title(Dof_names[i])

if figure_type_5:
    # initialisation de la figure pour detailed cost
    fig_5, axs_5 = plt.subplots(1, 1, tight_layout=True, figsize=(20, 15))
    axs_5.set_xticks(shift)
    axs_5.set_xticklabels(dynamics_types)
    axs_5.set_yscale("log")
    for k in range(28):
        axs_5.plot(-0.4, 0, color=colors[k], label=label_objectives[k])
    axs_5.legend()

# Initialisation des mesures de performance a comparer pour tous les essais et tous les types de dynamique dans l'ordre suivant:
# explicit, #root_explicit, #implicit, #root_implicit
angular_momentum_rmsd_all = np.zeros((6, 100))
linear_momentum_rmsd_all = np.zeros((6, 100))
residual_tau_rms_all = np.zeros((6, 100))
computation_time_all = np.zeros((6, 100))
cost_all = np.zeros((6, 100))
iterations_all = np.zeros((6, 100))
angular_momentum_rmsd_all[:] = np.nan
linear_momentum_rmsd_all[:] = np.nan
residual_tau_rms_all[:] = np.nan
computation_time_all[:] = np.nan
cost_all[:] = np.nan
iterations_all[:] = np.nan

variables_list = {
    "angular_momentum_rmsd_all": angular_momentum_rmsd_all,
    "linear_momentum_rmsd_all": linear_momentum_rmsd_all,
    "residual_tau_rms_all": residual_tau_rms_all,
    "computation_time_all": computation_time_all,
    "cost_all": cost_all,
    "iterations_all": iterations_all,
}

# parourir tous les fichiers de résultat du forlder
for file in os.listdir(out_path_raw):
    if file[-5:] == ".pckl":
        if "explicit" in file:
            if "root_" in file:
                i_dynamics_type = 1
            else:
                i_dynamics_type = 0
        if "implicit" in file:
            if "root_" in file:
                i_dynamics_type = 3
            else:
                i_dynamics_type = 2
        # idx_irand = file.find("i_rand") + 6 # pour les noms de fichiers qui comprennent i_rand au lieu de irand
        idx_irand = file.find("irand") + 5
        if idx_irand == 4:
            continue  # c'est que le fichier n'a pas le bon nom
        idx_pckl = file.find(".pckl")
        i_rand = int(file[idx_irand:idx_pckl])

        # remplir les mesures de performance pour l'essai en cours de lecture
        (
            variables_list["angular_momentum_rmsd_all"][i_dynamics_type, i_rand],
            variables_list["linear_momentum_rmsd_all"][i_dynamics_type, i_rand],
            variables_list["residual_tau_rms_all"][i_dynamics_type, i_rand],
            variables_list["computation_time_all"][i_dynamics_type, i_rand],
            variables_list["cost_all"][i_dynamics_type, i_rand],
            variables_list["iterations_all"][i_dynamics_type, i_rand],
            min_cost_varilables_values,
        ) = Analyses(
            out_path_raw,
            file,
            out_path_secondary_variables,
            i_rand,
            i_dynamics_type,
            axs,
            axs_5,
            min_cost_varilables_values,
        )

if figure_type_4:
    # ajout des lignes grasses des min cost sur les graphs de Q, Qdot, Qddot, Tau
    for i_dynamics_type in range(4):
        for i in range(15):
            axs[0][i].plot(
                min_cost_varilables_values["time_min"][i_dynamics_type],
                min_cost_varilables_values["q_min"][i_dynamics_type][i, :],
                "-",
                color=colors[i_dynamics_type],
                linewidth=4,
            )
            axs[1][i].plot(
                min_cost_varilables_values["time_min"][i_dynamics_type],
                min_cost_varilables_values["qdot_min"][i_dynamics_type][i, :],
                "-",
                color=colors[i_dynamics_type],
                linewidth=4,
            )
            axs[2][i].plot(
                min_cost_varilables_values["time_min"][i_dynamics_type],
                min_cost_varilables_values["qddot_min"][i_dynamics_type][i, :],
                "-",
                color=colors[i_dynamics_type],
                linewidth=4,
            )
            axs[3][i].step(
                min_cost_varilables_values["time_min"][i_dynamics_type],
                min_cost_varilables_values["tau_min"][i_dynamics_type][i, :],
                "-",
                color=colors[i_dynamics_type],
                linewidth=4,
            )

    if animation_min_cost:
        for i_dynamics_type in range(4):
            b = bioviz.Viz("/home/user/Documents/Programmation/Eve/OnDynamicsForSommersaults/Model_JeCh_15DoFs.bioMod")
            b.load_movement(min_cost_varilables_values["q_min"][i_dynamics_type])
            b.exec()

    plt.show()
    fig_1.savefig(f"Comparaison_Q.png")  # , dpi=900)v
    fig_2.savefig(f"Comparaison_Qdot.png")  # , dpi=900)
    fig_3.savefig(f"Comparaison_Qddot.png")  # , dpi=900)
    fig_4.savefig(f"Comparaison_Tau.png")  # , dpi=900)

if figure_type_5:
    plt.show()
    fig_5.savefig(f"Detailed_cost_type.png")  # , dpi=900)

# faire les analyses et graphiques de variables secondaires
graphs_analyses(variables_list)
