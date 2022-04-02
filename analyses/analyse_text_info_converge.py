"""
This script is used to print out information about the main results comparing the different dynamics
studying the effect of incrasing the number of shooting points.
It requires the dataframe of all results to run the script.
"""
import pandas as pd
from pandas import DataFrame

df_results = pd.read_pickle("Dataframe_convergence_metrics_5.pkl")

# Did everything converged ?
a = len(df_results[df_results["status"] == 1])
print(f"{a} of the trials did not converge to an optimal solutions")
dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[1, 3, 0, 2]]
for d in dyn:
    print(d)
    a = len(df_results[(df_results["status"] == 1) & (df_results["dynamics_type_label"] == d)])
    print(f"{a} of the trials with {d} did not converge to an optimal solutions")

# select only the one who converged
df_results = df_results[df_results["status"] == 0]


def batch_computation(df_results: DataFrame, key: str, d: str) -> float:
    """
    This function is used to compute the ratio between the min and the max for a given dynamics d and a given key.

    Parameters
    ----------
    df_results : DataFrame
        The dataframe containing the results.
    key : str
        The key suc as "computation_time" or "cost".
    d : str
        The dynamics type such as the one contained in dynamics_type_label.

    Returns
    -------
    ratio : float
        The ratio between the max and the min.
    """
    df = df_results[key]
    n_shoot_max = df_results["n_shooting_tot"][df_results["dynamics_type_label"] == d].max()
    df = df[df_results["dynamics_type_label"] == d]
    df_max = df[df_results["n_shooting_tot"] == n_shoot_max]
    df_min = df[df_results["n_shooting_tot"] == 150]

    t_mean_min = df_min.mean()
    t_mean_max = df_max.mean()

    return t_mean_max / t_mean_min


# Time and iterations
print(
    "For Full-Exp, Base-Exp, Full-Imp-$\qddot$, and Base-Imp-$\qddot$, Full-Imp-$\qdddot$, and Base-Imp-$\qdddot$, respectively"
    " the time (and iterations) required to converge"
    " were in average "
)
for ii, d in enumerate(dyn):
    df_results["computation_time"] = df_results["computation_time"] / 60
    ratio = batch_computation(df_results, "computation_time", d)
    print(f"time to converge was {ratio} time longer for {d}")

# Torque residuals
print("While implicit transcriptions were not consistent,")
print("Translation torque residuals respectively reached")
for ii, d in enumerate(dyn):
    ratio = batch_computation(df_results, "int_T", d)
    print(f"Translation torque residuals were {1/ratio} larger for {d}")
print(" ")
for ii, d in enumerate(dyn):
    ratio = batch_computation(df_results, "int_R", d)
    print(f"Rotation torque residuals were {1/ratio} larger for {d}")
print(" ")
for ii, d in enumerate(dyn):
    ratio = batch_computation(df_results, "linear_momentum_rmse", d)
    print(f"Linear momentum rmse were {1/ratio} larger for {d}")
print(" ")
for ii, d in enumerate(dyn):
    ratio = batch_computation(df_results, "angular_momentum_rmse", d)
    print(f"Angular momentum rmse were {1/ratio} larger for {d}")

# Costs
print("The average cost were")
for ii, d in enumerate(dyn):
    ratio = batch_computation(df_results, "cost_J", d)
    print(f"Cost were {1/ratio} larger for {d}")
    print(ratio)


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
    print(f"{t_mean} $\pm$ {t_std} \si{{kg.m^2.s}}, ")
print(".")
print("And it lead to RMSe in angular momentum of")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].mean(), 2)
    t_std = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].std(), 2)
    print(f"{t_mean} $\pm$ {t_std} \si{{kg.m.s^{{-1}} }}, ")
print(".")
