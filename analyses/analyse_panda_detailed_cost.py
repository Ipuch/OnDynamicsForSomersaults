from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

df_results = pd.read_pickle("Dataframe_results_metrics.pkl")

df_results["dynamics_type_label"] = None
df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "dynamics_type_label"] = r"$\text{Exp-Full}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dynamics_type_label"] = r"$\text{Exp-Base}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dynamics_type_label"] = r"$\text{Imp-Full-}\ddot{q}$"
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dynamics_type_label"] = r"$\text{Imp-Base-}\ddot{q}$"
df_results.loc[df_results[
                   "dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dynamics_type_label"] = r"$\text{Imp-Full-}\dddot{q}$"
df_results.loc[
    df_results[
        "dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dynamics_type_label"] = r"$\text{Imp-Base-}\dddot{q}$"

print("h")



dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[2, 4, 3, 5, 0, 1]]
dyn = dyn[2:]
grps = ["Explicit", "Explicit", "Implicit_qddot", "Implicit_qddot", "Implicit_qdddot", "Implicit_qdddot"]
grps = grps[2:]
colors = px.colors.qualitative.D3[2:]
print(dyn)