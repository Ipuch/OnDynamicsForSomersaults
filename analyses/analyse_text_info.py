"""
This script is used to print out information about the main results comparing the different dynamics.
It requires the dataframe of all results to run the script.
"""
import pandas as pd

df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")

# Did everything converged ?
a = len(df_results[df_results["status"] == 1])
print(f"{a} of the trials did not converge to an optimal solutions")
dyn = df_results["dynamics_type_label"].unique()
dyn = dyn[[0, 3, 2, 5, 1, 4]]

for d in dyn:
    print(d)
    a = len(df_results[(df_results["status"] == 1) & (df_results["dynamics_type_label"] == d)])
    print(f"{a} of the trials with {d} did not converge to an optimal solutions")

# select only the one who converged
df_results = df_results[df_results["status"] == 0]

# only the one that were in the main cluster
df_sub = df_results[df_results["main_cluster"] == True]
# Clusters
print("The number of optimal cost in the main cluster were respectively")
for ii, d in enumerate(dyn):
    df_sub_sub = df_sub[df_sub["dynamics_type_label"] == d]
    print(f"{df_sub_sub.__len__()}")
print("for explicit{}, rootexplicit{}, implicit{}, rootimplicit{}, implicitqdddot{} " "and rootimplicitqdddot{}. ")

# Costs
print("The average cost were")
for ii, d in enumerate(dyn):
    c_mean = round(df_results["cost_J"][df_results["dynamics_type_label"] == d].mean(), 1)
    c_std = round(df_results["cost_J"][df_results["dynamics_type_label"] == d].std(), 1)
    print(f"{c_mean} $\pm$ {c_std} min, ")
    c_median = round(df_results["cost_J"][df_results["dynamics_type_label"] == d].median(), 2)
    c_first_quartile = round(df_results["cost_J"][df_results["dynamics_type_label"] == d].quantile([0.25]).iloc[0], 2)
    c_second_quartile = round(df_results["cost_J"][df_results["dynamics_type_label"] == d].quantile([0.75]).iloc[0], 2)
    c_min = round(df_results["cost_J"][df_results["dynamics_type_label"] == d].min(), 2)
    c_max = round(df_results["cost_J"][df_results["dynamics_type_label"] == d].max(), 2)
    print(f"{c_median} ({c_first_quartile} - {c_second_quartile}, {c_min} - {c_max})")
    print(f"IC : {c_second_quartile - c_first_quartile}")
print(
    "for \explicit{}, \rootexplicit{}, \implicit{}, \rootimplicit{}, \implicitqdddot{} "
    "and \rootimplicitqdddot{} respectively (Fig.~\ref{fig:cost}). "
)

# Time and iterations
print(
    "For Full-Exp, Base-Exp, Full-Imp-$\qddot$, and Base-Imp-$\qddot$, Full-Imp-$\qdddot$, and Base-Imp-$\qdddot$, respectively"
    " the time (and iterations) required to converge"
    " were in average "
)
for ii, d in enumerate(dyn):
    t_mean = round(df_results["computation_time"][df_results["dynamics_type_label"] == d].mean() / 60, 1)
    t_std = round(df_results["computation_time"][df_results["dynamics_type_label"] == d].std() / 60, 1)
    iter = int(df_results["iterations"][df_results["dynamics_type_label"] == d].mean() / 60)
    print(f"{t_mean} $\pm$ {t_std} min, ")
    c_median = round(df_results["computation_time"][df_results["dynamics_type_label"] == d].median() / 60, 2)
    c_first_quartile = round(
        df_results["computation_time"][df_results["dynamics_type_label"] == d].quantile([0.25]).iloc[0] / 60, 2
    )
    c_second_quartile = round(
        df_results["computation_time"][df_results["dynamics_type_label"] == d].quantile([0.75]).iloc[0] / 60, 2
    )
    c_min = round(df_results["computation_time"][df_results["dynamics_type_label"] == d].min() / 60, 2)
    c_max = round(df_results["computation_time"][df_results["dynamics_type_label"] == d].max() / 60, 2)
    print(f"{c_median} ({c_first_quartile} - {c_second_quartile}, {c_min} - {c_max}) min")
    print(f"IC : {c_second_quartile - c_first_quartile}")

# iterations
print(
    "For Full-Exp, Base-Exp, Full-Imp-$\qddot$, and Base-Imp-$\qddot$, Full-Imp-$\qdddot$, and Base-Imp-$\qdddot$, respectively"
    " the iterations"
    " were in average "
)
for ii, d in enumerate(dyn):
    t_mean = round(df_results["iterations"][df_results["dynamics_type_label"] == d].mean(), 1)
    t_std = round(df_results["iterations"][df_results["dynamics_type_label"] == d].std(), 1)
    print(f"{int(t_mean)} $\pm$ {int(t_std)}, ")

# Torque residuals
print("While implicit transcriptions were not consistent,")
print("Translation torque residuals respectively reached")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["int_T"][df_results["dynamics_type_label"] == d].mean(), 1)
    t_std = round(df_results["int_T"][df_results["dynamics_type_label"] == d].std(), 1)
    print(f"{t_mean} $\pm$ {t_std} \si{{N.s}}, ")
    c_median = round(df_results["int_T"][df_results["dynamics_type_label"] == d].median(), 2)
    c_first_quartile = round(df_results["int_T"][df_results["dynamics_type_label"] == d].quantile([0.25]).iloc[0], 2)
    c_second_quartile = round(df_results["int_T"][df_results["dynamics_type_label"] == d].quantile([0.75]).iloc[0], 2)
    c_min = round(df_results["int_T"][df_results["dynamics_type_label"] == d].min(), 2)
    c_max = round(df_results["int_T"][df_results["dynamics_type_label"] == d].max(), 2)
    print(f"{c_median} ({c_first_quartile} - {c_second_quartile}, {c_min} - {c_max}) \si{{N.s}}")
    print(f"IC : {c_second_quartile - c_first_quartile}")
print(".")
print("Rotation torque residuals respectively reached")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["int_R"][df_results["dynamics_type_label"] == d].mean(), 1)
    t_std = round(df_results["int_R"][df_results["dynamics_type_label"] == d].std(), 1)
    print(f"{t_mean} $\pm$ {t_std} \si{{N.m.s}}, ")
    c_median = round(df_results["int_R"][df_results["dynamics_type_label"] == d].median(), 2)
    c_first_quartile = round(df_results["int_R"][df_results["dynamics_type_label"] == d].quantile([0.25]).iloc[0], 2)
    c_second_quartile = round(df_results["int_R"][df_results["dynamics_type_label"] == d].quantile([0.75]).iloc[0], 2)
    c_min = round(df_results["int_R"][df_results["dynamics_type_label"] == d].min(), 2)
    c_max = round(df_results["int_R"][df_results["dynamics_type_label"] == d].max(), 2)
    print(f"{c_median} ({c_first_quartile} - {c_second_quartile}, {c_min} - {c_max}) \si{{N.s}}")
    print(f"IC : {c_second_quartile - c_first_quartile}")
print(".")
print("It lead to RMSe in linear momentum of")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].mean(), 2)
    t_std = round(df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].std(), 2)
    print(f"{t_mean} $\pm$ {t_std} \si{{kg.m^2.s}}, ")
    c_median = round(df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].median(), 2)
    c_first_quartile = round(
        df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].quantile([0.25]).iloc[0], 2
    )
    c_second_quartile = round(
        df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].quantile([0.75]).iloc[0], 2
    )
    c_min = round(df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].min(), 2)
    c_max = round(df_results["linear_momentum_rmse"][df_results["dynamics_type_label"] == d].max(), 2)
    print(f"{c_median} ({c_first_quartile} - {c_second_quartile}, {c_min} - {c_max}) \si{{N.s}}")
    print(f"IC : {c_second_quartile - c_first_quartile}")
print(".")
print("And it lead to RMSe in angular momentum of")
for ii, d in enumerate(dyn[2:]):
    t_mean = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].mean(), 2)
    t_std = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].std(), 2)
    print(f"{t_mean} $\pm$ {t_std} \si{{kg.m.s^{{-1}} }}, ")
    c_median = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].median(), 2)
    c_first_quartile = round(
        df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].quantile([0.25]).iloc[0], 2
    )
    c_second_quartile = round(
        df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].quantile([0.75]).iloc[0], 2
    )
    c_min = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].min(), 2)
    c_max = round(df_results["angular_momentum_rmse"][df_results["dynamics_type_label"] == d].max(), 2)
    print(f"{c_median} ({c_first_quartile} - {c_second_quartile}, {c_min} - {c_max}) \si{{N.s}}")
    print(f"IC : {c_second_quartile - c_first_quartile}")
print(".")
