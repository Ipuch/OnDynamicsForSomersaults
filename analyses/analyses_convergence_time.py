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
import plotly.graph_objects as go

path_file = "../../OnDynamicsForSommersaults_results/raw_convergence_all"
model = "../Model_JeCh_15DoFs.bioMod"
# ouvrir les fichiers
files = os.listdir(path_file)
files.sort()

time_res = pd.DataFrame(columns=
                        ["time",
                         "n_shooting"])

dynamic_types = ["implicit", "root_implicit"]
for dynamic_type in dynamic_types:
    for i, file in enumerate(files):
        file_path = open(f"{path_file}/{file}", "rb")
        data = pickle.load(file_path)
        # print(file + "\n")
        # print(data["status"])
        if file.endswith(".pckl") and data["status"] == 0 and data["dynamics_type"] == dynamic_type:
            # if np.sum(data["n_shooting"]) == 480:
            #     print("hey")

            t = data["computation_time"]
            D = {"time": t,
                 "n_shooting": np.sum(data["n_shooting"]),
                 "dynamic_type": dynamic_type}
            time_res = time_res.append(D, ignore_index=True)

# plt.legend()
# plt.show()
dynamic_types = ["implicit", "root_implicit"]
residus_means_std = dict()
for dynamic_type in dynamic_types:
    residus_means_std[dynamic_type] = pd.DataFrame(columns=["time_mean",
                                                            "time_std",
                                                            "n_shooting"])
    sub_time_res = time_res[time_res["dynamic_type"] == dynamic_type]
    for n_shoot in time_res["n_shooting"].unique():
        sub_df = sub_time_res[sub_time_res["n_shooting"] == n_shoot]
        mean_t = np.mean(sub_df["time"])
        std_t = np.std(sub_df["time"])
        D = {"time_mean": mean_t,
             "time_std": std_t,
             "n_shooting": n_shoot}
        residus_means_std[dynamic_type] = residus_means_std[dynamic_type].append(D, ignore_index=True)

pal = sns.color_palette(palette="rocket_r", n_colors=2)

fig = go.Figure()

s = 13
article_names = ["Imp-Full", "Imp-Base"]
for jj, dynamic_type in enumerate(dynamic_types):
    sub_df = time_res[time_res["dynamic_type"] == dynamic_type]

    fig.add_scatter(cliponaxis=True, x=residus_means_std[dynamic_type]["n_shooting"],
                    y=residus_means_std[dynamic_type]["time_mean"],
                    # error_y=dict(
                    #     array=residus_means_std[dynamic_type]["time_std"],
                    #     thickness=5,
                    # ),
                    marker=dict(
                        color=f"rgb{str(pal[jj])}",
                        size=s
                    ),
                    mode='markers',
                    opacity=0.5,
                    marker_line_width=2,
                    legendgroup='group2',
                    legendgrouptitle_text="Mean and Standard deviation",
                    name=article_names[jj])

for jj, dynamic_type in enumerate(dynamic_types):
    sub_df = time_res[time_res["dynamic_type"] == dynamic_type]
    fig.add_scatter(x=sub_df["n_shooting"], y=sub_df["time"], mode='markers',
                    marker_color=f"rgb{str(pal[jj])}", name=article_names[jj],
                    legendgrouptitle_text="Simulation outputs",
                    legendgroup='group1')

# Update xaxis properties
fig.update_xaxes(title_text=r'$\textrm{Mesh point number}$', showline=True, linecolor='black',
                 ticks="outside")

# Update yaxis properties
fig.update_yaxes(title_text=r'$\textrm{time (}s\text{)}$', showline=True,
                 linecolor='black', ticks="outside", type="log")

fig.update_layout(height=400, width=600, paper_bgcolor='rgba(255,255,255,1)',
                  plot_bgcolor='rgba(255,255,255,1)',
                  legend=dict(
                      title_font_family="Times New Roman",
                      font=dict(
                          family="Times New Roman",
                          color="black",
                          size=12
                      ),
                      orientation="h",
                      xanchor="center",
                      x=0.5, y=-0.25),
                  font=dict(
                      size=12,
                      family="Times New Roman",
                  ),
                  xaxis=dict(color="black"),
                  yaxis=dict(color="black"),
                  template="simple_white"
                  )

fig.show()
