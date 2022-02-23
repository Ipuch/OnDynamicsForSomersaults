import os
import pickle
from utils import (
    stack_states,
    stack_controls,
    define_time,
    angular_momentum_deviation,
    angular_momentum_time_series,
    linear_momentum_time_series,
    linear_momentum_deviation,
    comdot_time_series,
    comddot_time_series,
    residual_torque_time_series,
    define_integrated_time,
    define_control_integrated,
)
import biorbd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

path_file = "../../OnDynamicsForSommersaults_results/raw_convergence_all"
model = "../Model_JeCh_15DoFs.bioMod"
# ouvrir les fichiers
files = os.listdir(path_file)
files.sort()

torque_res = pd.DataFrame(columns=
                          ["Torque Translation Residuals",
                           "Torque Rotation Residuals",
                           "Angular Momentum RMSE",
                           "Linear Momentum RMSE",
                           "n_shooting"])

dynamic_types = ["implicit", "root_implicit"]
for dynamic_type in dynamic_types:
    for i, file in enumerate(files):
        file_path = open(f"{path_file}/{file}", "rb")
        data = pickle.load(file_path)
        # print(file + "\n")
        # print(data["status"])
        if file.endswith(".pckl") and data["status"] == 0 and data["dynamics_type"] == dynamic_type:
            n_step = 5
            t_integrated = define_integrated_time(data["parameters"]["time"], data["n_shooting"], n_step)
            q_integrated = data["q_integrated"]
            qdot_integrated = data["qdot_integrated"]
            # tau_integrated = define_control_integrated(data["controls"], n_step, "tau")
            qddot_integrated = define_control_integrated(data["controls"], n_step, "qddot")
            m = biorbd.Model(model)

            T = residual_torque_time_series(m, q_integrated, qdot_integrated, qddot_integrated)[:3]
            # compute the norm and the integral of it
            T = np.linalg.norm(T, axis=0)
            int_T = np.zeros(1)
            for j in range(T.shape[0] - 1):
                dt = np.diff(t_integrated[j:j + 2])[0]
                if dt != 0:
                    int_T += np.trapz(T[j:j + 2], dx=dt)

            # to be done with rotations
            R = residual_torque_time_series(m, q_integrated, qdot_integrated, qddot_integrated)[3:]
            R = np.linalg.norm(R, axis=0)
            int_R = np.zeros(1)
            for j in range(R.shape[0] - 1):
                dt = np.diff(t_integrated[j:j + 2])[0]
                if dt != 0:
                    int_R += np.trapz(R[j:j + 2], dx=dt)

            # plt.figure(1)
            # plt.plot(t_integrated, R, label=str(np.sum(data["n_shooting"])))
            # plt.figure(2)
            # plt.plot(t,R)

            t = define_time(data["parameters"]["time"], data["n_shooting"])
            q = stack_states(data["states"], "q")
            qdot = stack_states(data["states"], "qdot")
            qddot = stack_controls(data["controls"], "qddot")
            # tau = stack_controls(data["controls"], "tau")

            angular_momentum = angular_momentum_time_series(m, q, qdot)
            linear_momentum = linear_momentum_time_series(m, q, qdot)
            comdot = comdot_time_series(m, q, qdot)
            comddot = comddot_time_series(m, q, qdot, qddot)
            mass = m.mass()
            angular_momentum_rmse = angular_momentum_deviation(angular_momentum)
            linear_momentum_rmse = linear_momentum_deviation(mass, comdot, t, comddot)
            # plt.figure(2)
            # plt.plot(t, np.linalg.norm(angular_momentum, axis=0), label=str(np.sum(data["n_shooting"])))

            D = {"Torque Translation Residuals": int_T[0],
                 "Torque Rotation Residuals": int_R[0],
                 "Angular Momentum RMSE": angular_momentum_rmse,
                 "Linear Momentum RMSE": linear_momentum_rmse,
                 "n_shooting": np.sum(data["n_shooting"]),
                 "dynamic_type": dynamic_type}
            torque_res = torque_res.append(D, ignore_index=True)

# plt.legend()
# plt.show()
dynamic_types = ["implicit", "root_implicit"]
residus_means_std = dict()
for dynamic_type in dynamic_types:
    residus_means_std[dynamic_type] = pd.DataFrame(columns=["trans_torque_mean",
                                                            "trans_torque_std",
                                                            "rot_torque_mean",
                                                            "rot_torque_std",
                                                            "angular_momentum_mean",
                                                            "angular_momentum_std",
                                                            "linear_momentum_mean",
                                                            "linear_momentum_std",
                                                            "n_shooting"])
    sub_torque_res = torque_res[torque_res["dynamic_type"] == dynamic_type]
    for n_shoot in torque_res["n_shooting"].unique():
        sub_df = sub_torque_res[sub_torque_res["n_shooting"] == n_shoot]
        mean_t = np.mean(sub_df["Torque Translation Residuals"])
        std_t = np.std(sub_df["Torque Translation Residuals"])
        mean_r = np.mean(sub_df["Torque Rotation Residuals"])
        std_r = np.std(sub_df["Torque Rotation Residuals"])
        mean_angular_momentum = np.mean(sub_df["Angular Momentum RMSE"])
        std_angular_momentum = np.std(sub_df["Angular Momentum RMSE"])
        mean_linear_momentum = np.mean(sub_df["Linear Momentum RMSE"])
        std_linear_momentum = np.std(sub_df["Linear Momentum RMSE"])
        D = {"trans_torque_mean": mean_t,
             "trans_torque_std": std_t,
             "rot_torque_mean": mean_r,
             "rot_torque_std": std_r,
             "mean_angular_momentum": mean_angular_momentum,
             "std_angular_momentum": std_angular_momentum,
             "mean_linear_momentum": mean_linear_momentum,
             "std_linear_momentum": std_linear_momentum,
             "n_shooting": n_shoot}
        residus_means_std[dynamic_type] = residus_means_std[dynamic_type].append(D, ignore_index=True)

pal = sns.color_palette(palette="rocket_r", n_colors=2)

fig = make_subplots(rows=2,
                    cols=2,
                    subplot_titles=(
                        r"$\textrm{Translation Torque Residuals}$", r"$\textrm{Rotation Torque Residuals}$",
                        r"$\textrm{Angular Momentum RMSe}$", r"$\textrm{Linear Momentum RMSe}$"))

s = 13
article_names = ["Imp-Full", "Imp-Base"]
for jj, dynamic_type in enumerate(dynamic_types):
    sub_df = torque_res[torque_res["dynamic_type"] == dynamic_type]

    fig.add_scatter(cliponaxis=True, x=residus_means_std[dynamic_type]["n_shooting"],
                    y=residus_means_std[dynamic_type]["trans_torque_mean"],
                    error_y=dict(
                        array=residus_means_std[dynamic_type]["trans_torque_std"],
                        thickness=5,
                    ),
                    marker=dict(
                        color=f"rgb{str(pal[jj])}",
                        size=s
                    ),
                    mode='markers',
                    opacity=0.5,
                    marker_line_width=2,
                    legendgroup='group2',
                    legendgrouptitle_text="Mean and Standard deviation",
                    name=article_names[jj],
                    row=1, col=1)
    fig.add_scatter(cliponaxis=True, x=residus_means_std[dynamic_type]["n_shooting"],
                    y=residus_means_std[dynamic_type]["rot_torque_mean"],
                    error_y=dict(
                        array=residus_means_std[dynamic_type]["rot_torque_std"],
                        thickness=5,
                    ),
                    marker=dict(
                        color=f"rgb{str(pal[jj])}",
                        size=s
                    ),
                    mode='markers',
                    opacity=0.5,
                    marker_line_width=2,
                    legendgroup='group2', showlegend=False,
                    legendgrouptitle_text="Mean and Standard deviation",
                    name=article_names[jj],
                    row=1, col=2)
    fig.add_scatter(cliponaxis=True, x=residus_means_std[dynamic_type]["n_shooting"],
                    y=residus_means_std[dynamic_type]["mean_angular_momentum"],
                    error_y=dict(
                        array=residus_means_std[dynamic_type]["std_angular_momentum"],
                        thickness=5,
                    ),
                    marker=dict(
                        color=f"rgb{str(pal[jj])}",
                        size=s
                    ),
                    mode='markers',
                    opacity=0.5,
                    marker_line_width=2,
                    legendgroup='group2', showlegend=False,
                    legendgrouptitle_text="Mean and Standard deviation",
                    name=article_names[jj],
                    row=2, col=1)

    fig.add_scatter(cliponaxis=True, x=residus_means_std[dynamic_type]["n_shooting"],
                    y=residus_means_std[dynamic_type]["mean_linear_momentum"],
                    error_y=dict(
                        array=residus_means_std[dynamic_type]["std_linear_momentum"],
                        thickness=5,
                    ),
                    marker=dict(
                        color=f"rgb{str(pal[jj])}",
                        size=s
                    ),
                    mode='markers',
                    opacity=0.5,
                    marker_line_width=2,
                    legendgroup='group2', showlegend=False,
                    legendgrouptitle_text="Mean and Standard deviation",
                    name=article_names[jj],
                    row=2, col=2)

for jj, dynamic_type in enumerate(dynamic_types):
    sub_df = torque_res[torque_res["dynamic_type"] == dynamic_type]
    fig.add_scatter(x=sub_df["n_shooting"], y=sub_df["Torque Translation Residuals"], mode='markers', row=1, col=1,
                    marker_color=f"rgb{str(pal[jj])}", name=article_names[jj],
                    legendgrouptitle_text="Simulation outputs",
                    legendgroup='group1')

    fig.add_scatter(x=sub_df["n_shooting"], y=sub_df["Torque Rotation Residuals"], mode='markers', row=1, col=2,
                    marker_color=f"rgb{str(pal[jj])}", name=article_names[jj],
                    legendgrouptitle_text="Simulation outputs",
                    showlegend=False, legendgroup='group1')

    fig.add_scatter(x=sub_df["n_shooting"], y=sub_df["Angular Momentum RMSE"], mode='markers', row=2, col=1,
                    marker_color=f"rgb{str(pal[jj])}", name=article_names[jj],
                    legendgrouptitle_text="Simulation outputs",
                    showlegend=False, legendgroup='group1')

    fig.add_scatter(x=sub_df["n_shooting"], y=sub_df["Linear Momentum RMSE"], mode='markers', row=2, col=2,
                    marker_color=f"rgb{str(pal[jj])}", name=article_names[jj],
                    legendgrouptitle_text="Simulation outputs",
                    showlegend=False, legendgroup='group1')

# Update xaxis properties
fig.update_xaxes(title_text=r'$\textrm{Mesh point number}$', row=1, col=1, showline=True, linecolor='black',
                 ticks="outside")
fig.update_xaxes(title_text=r'$\textrm{Mesh point number}$', row=1, col=2, showline=True, linecolor='black',
                 ticks="outside")
fig.update_xaxes(title_text=r'$\textrm{Mesh point number}$', row=2, col=1, showline=True, linecolor='black',
                 ticks="outside")
fig.update_xaxes(title_text=r'$\textrm{Mesh point number}$', row=2, col=2, showline=True, linecolor='black',
                 ticks="outside")

# Update yaxis properties
fig.update_yaxes(title_text=r'$\textrm{Absolute norm of torques (}N.s\text{)}$', row=1, col=1, showline=True,
                 linecolor='black', ticks="outside")
fig.update_yaxes(title_text=r'$\textrm{Absolute norm of torques (}N.m.s\text{)}$', row=1, col=2, showline=True,
                 linecolor='black', ticks="outside")
fig.update_yaxes(title_text=r'$\textrm{RMSe  (}N.m.s^{-1}\text{)}$', row=2, col=1, showline=True, linecolor='black',
                 ticks="outside")
fig.update_yaxes(title_text=r'$\textrm{RMSe  (}N.s^{-1}\text{)}$', row=2, col=2, showline=True, linecolor='black',
                 ticks="outside")

fig.update_layout(height=800, width=1200, paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  legend=dict(
                      title_font_family="Times New Roman",
                      font=dict(
                          family="Times New Roman",
                          color="black",
                          size=15
                      ),
                      orientation="h",
                      xanchor="center",
                      x=0.5, y=-0.15),
                  font=dict(
                      size=18,
                      family="Times New Roman",
                  ),
                  xaxis=dict(color="black"),
                  yaxis=dict(color="black"),
                  template="simple_white"
                  )

fig.show()
