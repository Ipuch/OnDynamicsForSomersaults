import sys
import os

sys.path.append(os.getcwd() + "/..")
from custom_dynamics.enums import MillerDynamics
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from IPython import embed

df_results = pd.read_pickle(
    "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/Donnees_Pierre/Dataframe_results_metrics.pkl"
)

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

# create a dataframe without explicit for the ANOVA on the dynamical consistency
df_results_implicit = df_results.drop(df_results[explicit_bool].index)
df_results_implicit.angular_momentum_rmse = pd.to_numeric(df_results_implicit.angular_momentum_rmse)
df_results_implicit.linear_momentum_rmse = pd.to_numeric(df_results_implicit.linear_momentum_rmse)
df_results_implicit.int_T = pd.to_numeric(df_results_implicit.int_T)


def function_test_anova(df_results, to_be_tested):
    model = ols(
        f"{to_be_tested} ~ C(Exp_Imp_Imp_jerk) + C(Root_Full) + C(Exp_Imp_Imp_jerk):C(Root_Full)", data=df_results
    ).fit()
    ANOVA_table = sm.stats.anova_lm(model, typ=2)
    print(ANOVA_table)
    return ANOVA_table


# def function_test_post_hoc(df_results, to_be_tested):
#     tukey_table = pairwise_tukeyhsd(endog=df_results[to_be_tested], groups=df_results['dynamics_type'], alpha=0.05)
#     print(tukey_table)
#     return tukey_table
#
# def show_stats_graphs(tukey_table):
#     rows = tukey_table.summary().data[1:]
#     plt.hlines(range(len(rows)), [row[4] for row in rows], [row[5] for row in rows])
#     plt.vlines(0, -1, len(rows) - 1, linestyles='dashed')
#     plt.gca().set_yticks(range(len(rows)))
#     plt.gca().set_yticklabels([f'{x[0]}-{x[1]}' for x in rows])
#     plt.show()
#     return

# def convert_from_pandas_to_array(df, type, to_be_tested):
#     print("to_do")
#     return "new format"


function_test_anova(df_results, "computation_time")
function_test_anova(df_results, "cost")

function_test_anova(df_results_implicit, "angular_momentum_rmse")
function_test_anova(df_results_implicit, "linear_momentum_rmse")
function_test_anova(df_results_implicit, "int_T")  # tau residuels


# function_test_post_hoc(df_results, "computation_time")