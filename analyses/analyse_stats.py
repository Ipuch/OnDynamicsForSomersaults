from custom_dynamics.enums import MillerDynamics
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

df_results = pd.read_pickle("Dataframe_results_metrics.pkl")

df_results["new_groups"] = None
my_bool = df_results["dynamics_type"] == (MillerDynamics.EXPLICIT or MillerDynamics.ROOT_EXPLICIT)
df_results.loc[my_bool, "new_groups"] = "EXPLICIT"

# Chi2
# two-way table
# stat.yale.edu/
# Courses/1997-98/
# 101/chisq.htm sur
# le nombre de solu-
# tion near optimal

def function_test_anova(df, type, to_be_test):

    print("to_do")
    return group_a_group, deux_a_deux


def convert_from_pandas_to_array(df, type, to_be_tested):
    print("to_do")
    return "new format"


p = function_test_anova(df, "dynamics_type", "computation_time"):
p = function_test_anova(df, "dynamics_type", "cost"):