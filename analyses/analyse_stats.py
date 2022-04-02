"""
This script does the main statistics analysis between each variables
"""
import sys
import os
from custom_dynamics.enums import MillerDynamics
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import matplotlib.pyplot as plt
from pandas import DataFrame


sys.path.append(os.getcwd() + "/..")
df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")

df_results["Exp_Imp_Imp_jerk"] = None
explicit_bool = np.logical_or(
    df_results["dynamics_type"] == MillerDynamics.EXPLICIT, df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT
)
df_results.loc[explicit_bool, "Exp_Imp_Imp_jerk"] = "EXPLICIT"
implicit_bool = np.logical_or(
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT, df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT
)
df_results.loc[implicit_bool, "Exp_Imp_Imp_jerk"] = "IMPLICIT"
implicit_qdddot_bool = np.logical_or(
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT,
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT,
)
df_results.loc[implicit_qdddot_bool, "Exp_Imp_Imp_jerk"] = "IMPLICIT_QDDDOT"

df_results["Root_Full"] = None
full_bool = np.logical_or(
    np.logical_or(
        df_results["dynamics_type"] == MillerDynamics.EXPLICIT, df_results["dynamics_type"] == MillerDynamics.IMPLICIT
    ),
    df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT,
)
df_results.loc[full_bool, "Root_Full"] = "FULL_BODY"
root_bool = np.logical_or(
    np.logical_or(
        df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT,
        df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT,
    ),
    df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT,
)
df_results.loc[root_bool, "Root_Full"] = "ROOT"

df_results["dynamics_type_label"] = None
for e in MillerDynamics:
    df_results.loc[df_results["dynamics_type"] == e, "dynamics_type_label"] = e.value

# create a dataframe without explicit for the ANOVA on the dynamical consistency
df_results_implicit = df_results.drop(df_results[explicit_bool].index)
df_results_implicit.angular_momentum_rmse = pd.to_numeric(df_results_implicit.angular_momentum_rmse)
df_results_implicit.linear_momentum_rmse = pd.to_numeric(df_results_implicit.linear_momentum_rmse)
df_results_implicit.int_T = pd.to_numeric(df_results_implicit.int_T)


def function_test_anova(df_results: DataFrame, to_be_tested: str):
    """
    This function does the two-ways ANOVA test on the dataframe for a given key to be tested

    Parameters
    ----------
    df_results : DataFrame
        The dataframe with the results
    to_be_tested : str
        The key to be tested

    Returns
    -------
    anova_results : DataFrame
        The results of the two-ways ANOVA test
    """
    df_results[to_be_tested] = pd.to_numeric(df_results[to_be_tested])
    model = ols(
        f"{to_be_tested} ~ C(Exp_Imp_Imp_jerk) + C(Root_Full) + C(Exp_Imp_Imp_jerk):C(Root_Full)", data=df_results
    ).fit()
    ANOVA_table = sm.stats.anova_lm(model, typ=2)
    print(ANOVA_table)
    return ANOVA_table


def function_test_post_hoc(df_results: DataFrame, to_be_tested: str):
    """
    This fnunction does the pairwise tukeyhsd test on the dataframe for a given key to be tested

    Parameters
    ----------
    df_results : DataFrame
        The dataframe with the results
    to_be_tested : str
        The key to be tested

    Returns
    -------
    post_hoc_results : DataFrame
        The results of the pairwise tukeyhsd test
    """
    tukey_table = pairwise_tukeyhsd(
        endog=df_results[to_be_tested], groups=df_results["dynamics_type_label"], alpha=0.05
    )
    print(tukey_table)
    return tukey_table


# def show_stats_graphs(tukey_table):
#     rows = tukey_table.summary().data[1:]
#     plt.hlines(range(len(rows)), [row[4] for row in rows], [row[5] for row in rows])
#     plt.vlines(0, -1, len(rows) - 1, linestyles="dashed")
#     plt.gca().set_yticks(range(len(rows)))
#     plt.gca().set_yticklabels([f"{x[0]}-{x[1]}" for x in rows])
#     plt.show()


print("Computation time")
res = function_test_anova(df_results, "computation_time")
post_hoc_res = function_test_post_hoc(df_results, "computation_time")
print("cost_J")
res = function_test_anova(df_results, "cost_J")
post_hoc_res = function_test_post_hoc(df_results, "cost_J")

print("angular_momentum_rmse")
res = function_test_anova(df_results_implicit, "angular_momentum_rmse")
post_hoc_res = function_test_post_hoc(df_results_implicit, "angular_momentum_rmse")
print("linear_momentum_rmse")
res = function_test_anova(df_results_implicit, "linear_momentum_rmse")
post_hoc_res = function_test_post_hoc(df_results_implicit, "linear_momentum_rmse")
print("int_T")
res = function_test_anova(df_results_implicit, "int_T")
post_hoc_res = function_test_post_hoc(df_results_implicit, "int_T")
print("int_R")
res = function_test_anova(df_results_implicit, "int_R")
post_hoc_res = function_test_post_hoc(df_results_implicit, "int_R")
