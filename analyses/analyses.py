import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import biorbd
import os
import pickle
import bioviz


def graphs_analyses(
    angular_momentum_rmsd_all,
    linear_momentum_rmsd_all,
    residual_tau_sum_all,
    computation_time_all,
    cost_all,
    iterations_all,
):

    figure_type_1 = True  # True  # Tous sur le meme,
    figure_type_2 = True  # True  # Tous sur le meme avec lignes qui relient les points
    figure_type_3 = True  # True  # Séparés

    colors = ["#2E5A90FF", "#00BA87FF", "#DDEA00FF", "#BE2AD0FF", "#76DF1FFF", "#13BBF2FF", "#500375FF"]
    shift = [-0.3, -0.1, 0.1, 0.3]
    labels = [
        "Angular\nMomentum\nRMSD",
        "Linear\nMomentum\nRMSD",
        "Resirual\nTau",
        "Computation\nTime",
        "Optimal\nCost",
        "Number\nof\niterations",
    ]
    dynamics_type = ["Explicit", "Root explicit", "Implicit", "Root Implicit"]

    weights = np.array([1e-1, 1, 1, 1, 1e-9, 1])

    variables_list = np.array(
        [
            angular_momentum_rmsd_all,
            linear_momentum_rmsd_all,
            residual_tau_sum_all,
            computation_time_all,
            cost_all,
            iterations_all,
        ]
    )

    variables_list_weighted = np.array([variables_list[j] * weights[j] for j in range(6)])

    variables_mean_list = np.zeros((6, 4))
    variables_std_list = np.zeros((6, 4))
    variables_mean_list_weighted = np.zeros((6, 4))
    variables_std_list_weighted = np.zeros((6, 4))
    for j in range(6):
        variables_mean_list[j, :] = np.nanmean(variables_list[j, :, :], axis=1)
        variables_std_list[j, :] = np.nanstd(variables_list[j, :, :], axis=1)
        variables_mean_list_weighted[j, :] = np.nanmean(variables_list_weighted[j, :, :], axis=1)
        variables_std_list_weighted[j, :] = np.nanstd(variables_list_weighted[j, :, :], axis=1)

    if figure_type_1:
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(labels)

        for j in range(6):
            for i in range(4):
                ax.plot(
                    np.array([j + shift[i] - 0.05, +shift[i] + 0.05]),
                    np.ones(
                        2,
                    )
                    * variables_mean_list_weighted[j, i],
                    color=colors[i],
                    linewidth=2,
                )
                if j == 0:
                    ax.bar(
                        j + shift[i],
                        2 * variables_std_list_weighted[j, i],
                        width=0.1,
                        color=colors[i],
                        bottom=variables_mean_list_weighted[j, i] - variables_std_list_weighted[j, i],
                        label=dynamics_type[i] + " mean $\pm$ std",
                        alpha=0.3,
                    )
                    ax.plot(
                        np.ones((100,)) * (j + shift[i]),
                        variables_list_weighted[j, i, :],
                        ".",
                        color=colors[i],
                        label=dynamics_type[i],
                    )
                else:
                    ax.bar(
                        j + shift[i],
                        2 * variables_std_list_weighted[j, i],
                        width=0.1,
                        color=colors[i],
                        bottom=variables_mean_list_weighted[j, i] - variables_std_list_weighted[j, i],
                        alpha=0.3,
                    )
                    ax.plot(np.ones((100,)) * (j + shift[i]), variables_list_weighted[j, i, :], ".", color=colors[i])

        plt.legend(
            loc="upper center",
            frameon=False,
            ncol=2,
            # fontsize=12,
            bbox_to_anchor=(0.5, 1.5),
        )

        plt.show()
        plt.savefig("Comparaison.png", dpi=900)

    if figure_type_2:
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(labels)

        for k in range(100):
            for i in range(4):
                if k == 0:
                    ax.plot(
                        np.arange(6) + shift[i],
                        variables_list_weighted[:, i, k],
                        "-",
                        marker=".",
                        color=colors[i],
                        linewidth=0.5,
                        label=dynamics_type[i],
                    )
                else:
                    ax.plot(
                        np.arange(6) + shift[i],
                        variables_list_weighted[:, i, k],
                        "-",
                        marker=".",
                        color=colors[i],
                        linewidth=0.5,
                    )

        for j in range(6):
            for i in range(4):
                if j == 0:
                    ax.bar(
                        j + shift[i],
                        2 * variables_std_list_weighted[j, i],
                        width=0.1,
                        color=colors[i],
                        bottom=variables_mean_list_weighted[j, i] - variables_std_list_weighted[j, i],
                        label=dynamics_type[i] + " mean $\pm$ std",
                        alpha=0.3,
                    )
                else:
                    ax.bar(
                        j + shift[i],
                        2 * variables_std_list_weighted[j, i],
                        width=0.1,
                        color=colors[i],
                        bottom=variables_mean_list_weighted[j, i] - variables_std_list_weighted[j, i],
                        alpha=0.3,
                    )

        plt.legend(
            loc="upper center",
            frameon=False,
            ncol=2,
            # fontsize=12,
            bbox_to_anchor=(0.5, 1.5),
        )
        plt.show()
        plt.savefig("Comparaison_lignes.png", dpi=900)

    if figure_type_3:
        # fig, axs = plt.subplots(2, 3, tight_layout=True)
        # axs = axs.ravel()

        for j in range(6):
            plt.figure()
            plt.xticks(shift, labels=dynamics_type)

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
                    plt.scatter(
                        np.ones((100,)) * shift[i], variables_list[j, i, :], c=np.linspace(0, 1, 100), cmap="viridis"
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
                    plt.scatter(
                        np.ones((100,)) * shift[i], variables_list[j, i, :], c=np.linspace(0, 1, 100), cmap="viridis"
                    )

            plt.legend(
                loc="upper center",
                frameon=False,
                ncol=2,
                # fontsize=12,
                bbox_to_anchor=(0.5, 1.5),
            )
            if labels[j] != "Optimal\nCost":
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
    figure_type_4,
    time_min,
    q_min,
    qdot_min,
    qddot_min,
    tau_min,
    min_cost,
):

    file = open(f"{out_path_raw}/{file_name}", "rb")
    data = pickle.load(file)

    q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))
    qdot = np.hstack((data["states"][0]["qdot"], data["states"][1]["qdot"]))
    t = data["parameters"]["time"]
    time = np.hstack(
        (
            np.linspace(0, float(t[0]), np.shape(data["states"][0]["q"])[1]),
            np.linspace(float(t[0]), float(t[0]) + float(t[1]), np.shape(data["states"][1]["q"])[1]),
        )
    )  # [1:]
    N = np.shape(q)[1]
    q_integrated = data["q_integrated"]
    qdot_integrated = data["qdot_integrated"]
    N_integrated = np.shape(q_integrated)[1]
    dynamics_type = data["dynamics_type"]
    computation_time = data["computation_time"]
    cost = data["cost"]
    iterations = data["iterations"]

    if dynamics_type == "explicit" or dynamics_type == "implicit":
        tau = np.hstack((data["controls"][0]["tau"], data["controls"][1]["tau"]))
        tau_integrated = np.hstack(
            (
                np.repeat(data["controls"][0]["tau"], 6, axis=1)[:, :-5],
                np.repeat(data["controls"][1]["tau"], 6, axis=1)[:, :-5],
            )
        )
    elif dynamics_type == "root_implicit":
        print(i_rand)
        print(i_dynamics_type)
        qddot = np.hstack((data["controls"][0]["qddot"], data["controls"][1]["qddot"]))
        qddot_integrated = np.hstack(
            (
                np.repeat(data["controls"][0]["qddot"], 6, axis=1)[:, :-5],
                np.repeat(data["controls"][1]["qddot"], 6, axis=1)[:, :-5],
            )
        )
    elif dynamics_type == "root_explicit":
        print(i_rand)
        print(i_dynamics_type)
        qddot = np.hstack((data["controls"][0]["qddot_joint"], data["controls"][1]["qddot_joint"]))
        qddot_integrated = np.hstack(
            (
                np.repeat(data["controls"][0]["qddot_joint"], 6, axis=1)[:, :-5],
                np.repeat(data["controls"][1]["qddot_joint"], 6, axis=1)[:, :-5],
            )
        )

    m = biorbd.Model("../" + data["model_path"])

    if dynamics_type == "root_explicit" or dynamics_type == "root_implicit":
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

    else:
        residual_tau_integrated = 0
        qddot = np.zeros((m.nbQ(), N))
        for node in range(N):
            qddot[:, node] = m.ForwardDynamics(
                q[:, node], qdot[:, node], np.hstack((np.zeros((6,)), tau[:, node]))
            ).to_array()
        tau = np.vstack((np.zeros((6, N)), tau))

    residual_tau_sum = np.nansum(np.abs(residual_tau_integrated))

    angular_momentum = np.zeros((3, N))
    angular_momentum_norm = np.zeros((N,))
    linear_momentum = np.zeros((3, N))
    CoM_position = np.zeros((3, N))
    CoM_velocity = np.zeros((3, N))
    CoM_acceleration = np.zeros((3, N))
    for node in range(N):
        angular_momentum[:, node] = m.angularMomentum(q[:, node], qdot[:, node], True).to_array()
        angular_momentum_norm[node] = np.linalg.norm(angular_momentum[:, node])
        linear_momentum[:, node] = m.CoMdot(q[:, node], qdot[:, node], True).to_array() * m.mass()
        CoM_position[:, node] = m.CoM(q[:, node], True).to_array()
        CoM_velocity[:, node] = m.CoMdot(q[:, node], qdot[:, node], True).to_array()
        CoM_acceleration[:, node] = m.CoMddot(q[:, node], qdot[:, node], qddot[:, node], True).to_array()

    angular_momentum_mean = np.mean(angular_momentum, axis=1)
    angular_momentum_rmsd = np.zeros((3,))
    linear_momentum_rmsd = np.zeros((3,))
    for i in range(3):
        angular_momentum_rmsd[i] = np.sqrt(((angular_momentum[i, :] - angular_momentum[i, 0]) ** 2).mean())
        if i == 0 or i == 1:
            linear_momentum_rmsd[i] = m.mass() * np.sqrt(((CoM_velocity[i, :] - CoM_velocity[i, 0]) ** 2).mean())
        else:
            linear_momentum_rmsd[i] = m.mass() * np.sqrt(
                ((CoM_velocity[i, :] - (CoM_acceleration[i, 0] * time + CoM_velocity[i, 0])) ** 2).mean()
            )

    f = open(f"{out_path_secondary_variables}/miller_{dynamics_type}_irand{i_rand}_analyses.pckl", "wb")
    data_secondary = {
        "angular_momentum": angular_momentum,
        "angular_momentum_norm": angular_momentum_norm,
        "linear_momentum": linear_momentum,
        "CoM_position": CoM_position,
        "residual_tau_integrated ": residual_tau_integrated,
        "angular_momentum_rmsd": angular_momentum_rmsd,
        "linear_momentum_rmsd": linear_momentum_rmsd,
        "residual_tau_sum": residual_tau_sum,
    }

    pickle.dump(data_secondary, f)
    f.close()

    colors = ["#2E5A90FF", "#00BA87FF", "#DDEA00FF", "#BE2AD0FF", "#76DF1FFF", "#13BBF2FF", "#500375FF"]
    if figure_type_4:
        for i in range(15):
            axs[0][i].plot(time, q[i, :], "-", color=colors[i_dynamics_type])
            axs[1][i].plot(time, qdot[i, :], "-", color=colors[i_dynamics_type])

            if i_dynamics_type == 1 and i < 9:
                axs[2][i + 6].plot(time, qddot[i, :], "-", color=colors[i_dynamics_type])
            elif i_dynamics_type != 1:
                axs[2][i].plot(time, qddot[i, :], "-", color=colors[i_dynamics_type])

            axs[3][i].step(time, tau[i, :], "-", color=colors[i_dynamics_type])

    if cost < min_cost[i_dynamics_type]:
        time_min[i_dynamics_type] = time
        q_min[i_dynamics_type] = q
        qdot_min[i_dynamics_type] = qdot
        qddot_min[i_dynamics_type] = qddot
        tau_min[i_dynamics_type] = tau
        min_cost[i_dynamics_type] = cost

    return (
        np.sum(np.abs(angular_momentum_rmsd)),
        np.sum(np.abs(linear_momentum_rmsd)),
        np.sum(residual_tau_sum),
        computation_time,
        cost,
        iterations,
        time_min,
        q_min,
        qdot_min,
        qddot_min,
        tau_min,
        min_cost,
    )


# starting of the function
out_path_raw = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw"
# out_path_raw = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw"
out_path_secondary_variables = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/secondary_variables"
# out_path_secondary_variables = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/secondary_variables"
animation_min_cost = False


min_cost = np.ones((4,)) * 1e30
# prefill the minimum optimal data
q_min = [[], [], [], []]
qdot_min = [[], [], [], []]
qddot_min = [[], [], [], []]
tau_min = [[], [], [], []]
time_min = [[], [], [], []]

figure_type_4 = True  # True  # Techniques
axs = []
if figure_type_4:
    fig_1, axs_1 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Q
    axs_1 = axs_1.ravel()
    fig_2, axs_2 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Qdot
    axs_2 = axs_2.ravel()
    fig_3, axs_3 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Qddot
    axs_3 = axs_3.ravel()
    fig_4, axs_4 = plt.subplots(5, 3, tight_layout=True, figsize=(20, 15))  # Tau
    axs_4 = axs_4.ravel()

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
    axs = [axs_1, axs_2, axs_3, axs_4]
    for ax in axs:
        for i in range(15):
            ax[i].set_title(Dof_names[i])


# explicit, #root_explicit, #implicit, #root_implicit
angular_momentum_rmsd_all = np.zeros((4, 100))
linear_momentum_rmsd_all = np.zeros((4, 100))
residual_tau_sum_all = np.zeros((4, 100))
computation_time_all = np.zeros((4, 100))
cost_all = np.zeros((4, 100))
iterations_all = np.zeros((4, 100))
angular_momentum_rmsd_all[:] = np.nan
linear_momentum_rmsd_all[:] = np.nan
residual_tau_sum_all[:] = np.nan
computation_time_all[:] = np.nan
cost_all[:] = np.nan
iterations_all[:] = np.nan

# To be removed
angular_momentum_rmsd_all[:, 0] = 0
linear_momentum_rmsd_all[:, 0] = 0
residual_tau_sum_all[:, 0] = 0
computation_time_all[:, 0] = 0
cost_all[:, 0] = 0
iterations_all[:, 0] = 0

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
        idx_irand = file.find("irand") + 5
        idx_pckl = file.find(".pckl")
        i_rand = int(file[idx_irand:idx_pckl])
        (
            angular_momentum_rmsd_all[i, i_rand],
            linear_momentum_rmsd_all[i, i_rand],
            residual_tau_sum_all[i, i_rand],
            computation_time_all[i, i_rand],
            cost_all[i, i_rand],
            iterations_all[i, i_rand],
            time_min,
            q_min,
            qdot_min,
            qddot_min,
            tau_min,
            min_cost,
        ) = Analyses(
            out_path_raw,
            file,
            out_path_secondary_variables,
            i_rand,
            i,
            axs,
            figure_type_4,
            time_min,
            q_min,
            qdot_min,
            qddot_min,
            tau_min,
            min_cost,
        )

if figure_type_4:
    colors = ["#224168FF", "#058979FF", "#B3BD00FF", "#851D91FF", "#509716FF"]
    for i_dynamics_type in [0]:
        for i in range(15):
            axs[0][i].plot(
                time_min[i_dynamics_type], q_min[i_dynamics_type][i, :], "-", color=colors[i_dynamics_type], linewidth=4
            )
            axs[1][i].plot(
                time_min[i_dynamics_type],
                qdot_min[i_dynamics_type][i, :],
                "-",
                color=colors[i_dynamics_type],
                linewidth=4,
            )
            axs[2][i].plot(
                time_min[i_dynamics_type],
                qddot_min[i_dynamics_type][i, :],
                "-",
                color=colors[i_dynamics_type],
                linewidth=4,
            )
            axs[3][i].step(
                time_min[i_dynamics_type],
                tau_min[i_dynamics_type][i, :],
                "-",
                color=colors[i_dynamics_type],
                linewidth=4,
            )

    if animation_min_cost:
        b = bioviz.Viz("/home/user/Documents/Programmation/Eve/OnDynamicsForSommersaults/Model_JeCh_15DoFs.bioMod")
        b.load_movement(q_min[0])
        b.exec()

    plt.show()
    fig_1.savefig(f"Comparaison_Q.png")  # , dpi=900)
    fig_2.savefig(f"Comparaison_Qdot.png")  # , dpi=900)
    fig_3.savefig(f"Comparaison_Qddot.png")  # , dpi=900)
    fig_4.savefig(f"Comparaison_Tau.png")  # , dpi=900)

graphs_analyses(
    angular_momentum_rmsd_all,
    linear_momentum_rmsd_all,
    residual_tau_sum_all,
    computation_time_all,
    cost_all,
    iterations_all,
)
