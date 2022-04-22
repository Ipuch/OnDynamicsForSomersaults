"""
This script does the main statistics analysis between each variables
It requires the dataframe of all results to run the script.
"""
import sys
import os
from custom_dynamics.enums import MillerDynamics
import pandas as pd
import numpy as np
from scipy import stats
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
df_results_implicit.int_R = pd.to_numeric(df_results_implicit.int_R)
df_results.cost_J = pd.to_numeric(df_results.cost_J)


def friedman_test_implicit(df_results: DataFrame, key: str):
    """
    This function performs Friedman test on the dataframe for a given key to be tested

    Parameters
    ----------
    df_results : DataFrame
        The dataframe with the results
    key : str
        The key to be tested

    Returns
    -------
    friedman_results : DataFrame
        The results of the Friedman test
    """
    group1 = df_results[df_results["dynamics_type"] == MillerDynamics.IMPLICIT][key].to_numpy()
    group2 = df_results[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT][key].to_numpy()
    group3 = df_results[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT][key].to_numpy()
    group4 = df_results[df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT][key].to_numpy()

    res = stats.friedmanchisquare(group1, group2, group3, group4)
    print(res)

    print(stats.mannwhitneyu(group1, group2))
    print(stats.mannwhitneyu(group1, group3))
    print(stats.mannwhitneyu(group1, group4))
    print(stats.mannwhitneyu(group2, group3))
    print(stats.mannwhitneyu(group2, group4))
    print(stats.mannwhitneyu(group3, group4))

    return res


friedman_test_implicit(df_results_implicit, "angular_momentum_rmse")
friedman_test_implicit(df_results_implicit, "linear_momentum_rmse")
friedman_test_implicit(df_results_implicit, "int_T")
friedman_test_implicit(df_results_implicit, "int_R")


def friedman_test(df_results: DataFrame, key: str):
    """
    This function performs Friedman test on the dataframe for a given key to be tested

    Parameters
    ----------
    df_results : DataFrame
        The dataframe with the results
    key : str
        The key to be tested

    Returns
    -------
    friedman_results : DataFrame
        The results of the Friedman test
    """
    group1 = df_results[df_results["dynamics_type"] == MillerDynamics.EXPLICIT][key].to_numpy()
    group2 = df_results[df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT][key].to_numpy()
    group3 = df_results[df_results["dynamics_type"] == MillerDynamics.IMPLICIT][key].to_numpy()
    group4 = df_results[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT][key].to_numpy()
    group5 = df_results[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT][key].to_numpy()
    group6 = df_results[df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT][key].to_numpy()

    res = stats.friedmanchisquare(group1, group2, group3, group4, group5, group6)
    print(res)

    print(stats.mannwhitneyu(group1, group2))
    print(stats.mannwhitneyu(group1, group3))
    print(stats.mannwhitneyu(group1, group4))
    print(stats.mannwhitneyu(group1, group5))
    print(stats.mannwhitneyu(group1, group6))
    print(stats.mannwhitneyu(group2, group3))
    print(stats.mannwhitneyu(group2, group4))
    print(stats.mannwhitneyu(group2, group5))
    print(stats.mannwhitneyu(group2, group6))
    print(stats.mannwhitneyu(group3, group4))
    print(stats.mannwhitneyu(group3, group5))
    print(stats.mannwhitneyu(group3, group6))
    print(stats.mannwhitneyu(group4, group5))
    print(stats.mannwhitneyu(group4, group6))
    print(stats.mannwhitneyu(group5, group6))

    return res


friedman_test(df_results, "cost_J")
friedman_test(df_results, "computation_time")
