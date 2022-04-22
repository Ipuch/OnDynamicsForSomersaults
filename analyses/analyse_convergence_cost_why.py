"""
This script plots all generalized coordinates and all torques for all MillerDynamics
contained in the main cluster of optimal costs
It requires the dataframe of all results to run the script.
"""

from custom_dynamics.enums import MillerDynamics
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import biorbd

df_results = pd.read_pickle("Dataframe_convergence_metrics_5.pkl")
out_path_file = "../../OnDynamicsForSommersaults_results/figures/V5"

df_results = df_results[df_results["status"] == 0]

for d in MillerDynamics:
    if d == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT or d == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        sub_df = df_results[df_results["dynamics_type"] == d]

        data = []
        n_shooting_tot = sub_df["n_shooting_tot"].unique()
        for n_shoot in n_shooting_tot:
            sub_df_n_shoot = sub_df[sub_df["n_shooting_tot"] == n_shoot]
            for i in range(0, 10):
                data.append(go.Bar(name=f"cost_J{i}", x=[n_shoot], y=[sub_df_n_shoot[f"cost_J{i}"].mean()]
                                   ,legendgroup=f"cost_J{i}"))

fig = go.Figure(data=data)
fig.show()


for d in MillerDynamics:
    if d == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT or d == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        sub_df = df_results[df_results["dynamics_type"] == d]

        fig = px.bar(sub_df, x="n_shooting_tot", y="cost")
        fig.show()
