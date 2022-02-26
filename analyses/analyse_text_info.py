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
    df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dynamics_type_label"
] = r"$\text{Exp-Base}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dynamics_type_label"
] = r"$\text{Imp-Full-}\ddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dynamics_type_label"
] = r"$\text{Imp-Base-}\ddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dynamics_type_label"
] = r"$\text{Imp-Full-}\dddot{q}$"
df_results.loc[
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dynamics_type_label"
] = r"$\text{Imp-Base-}\dddot{q}$"

# Did everything converged ?
a = len(df_results[df_results["status"] == 1])
print(f"{a} of the trials did not converge to an optimal solutions")
dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[2, 4, 3, 5, 0, 1]]
for d in dyn:
    print(d)
    a = len(df_results[(df_results["status"] == 1) & (df_results["dynamics_type_label"] == d)])
    print(f"{a} of the trials with {d} did not converge to an optimal solutions")

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

# Time and iterations
print(
    "For Exp-Full, Expl-Base, Imp-Full-qddot, and Imp-Base-qddot, Imp-Full-qdddot, and Imp-Base-qdddot, respectively"
    " the time (and iterations) required to converge"
    " were in average "
)
for ii, d in enumerate(dyn):
    t_mean = round(df_results["computation_time"][df_results["dynamics_type_label"] == d].mean() / 60, 1)
    t_std = round(df_results["computation_time"][df_results["dynamics_type_label"] == d].std() / 60, 1)
    iter = int(df_results["iterations"][df_results["dynamics_type_label"] == d].mean())
    print(f"{t_mean} $\pm$ {t_std} min ({iter} iterations in average), ")

# Torque residuals
print("While implicit transcriptions were not consistent,")
print("Translation torque residuals respectively reached")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["int_T"][df_results["dynamics_type_label"] == d].mean(), 1)
    t_std = round(df_results["int_T"][df_results["dynamics_type_label"] == d].std(), 1)
    print(f"{t_mean} $\pm$ {t_std} \si{{N.s}}, ")
print(".")
print("Rotation torque residuals respectively reached")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["int_R"][df_results["dynamics_type_label"] == d].mean(), 1)
    t_std = round(df_results["int_R"][df_results["dynamics_type_label"] == d].std(), 1)
    print(f"{t_mean} $\pm$ {t_std} \si{{N.m.s}}, ")
print(".")
print("It lead to RMSe in linear momentum of")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].mean(), 2)
    t_std = round(df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].std(), 2)
    print(f"{t_mean} $\pm$ {t_std} \si{{N.s^{{-1}} }}, ")
print(".")
print("And it lead to RMSe in angular momentum of")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].mean(), 2)
    t_std = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].std(), 2)
    print(f"{t_mean} $\pm$ {t_std} \si{{N.s^{{-1}} }}, ")
print(".")
