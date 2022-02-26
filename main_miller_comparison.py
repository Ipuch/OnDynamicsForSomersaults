import os, shutil
from typing import Union
# from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import date
import smtplib, ssl
import miller_run
from bioptim import OdeSolver
from custom_dynamics.enums import MillerDynamics

Date = date.today()
Date = Date.strftime("%d-%m-%y")

out_path_raw = "../OnDynamicsForSommersaults_results/raw_" + Date
try:
    os.mkdir(out_path_raw)
except:
    print("../OnDynamicsForSommersaults_results/raw_" + Date + " is already created ")

out_path_secondary_variables = "../OnDynamicsForSommersaults_results/secondary_variables"

cpu_number = cpu_count()

# dynamics_types = ["explicit", "root_explicit", "implicit", "root_implicit"]
# ode_solver = [OdeSolver.RK4, OdeSolver.RK4, OdeSolver.RK2, OdeSolver.RK2]

model_str = "Model_JeCh_15DoFs.bioMod"
n_shooting = (125, 25)
nstep = 5

n_threads = 1  # Should be 8
ode_solver = [OdeSolver.RK4, OdeSolver.RK4]
dynamics_types = [MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, MillerDynamics.ROOT_IMPLICIT_QDDDOT]


def generate_calls(
        n: Union[int, list],
        Date,
        n_shooting: tuple,
        dynamics_types: list,
        ode_solver: list,
        nstep: int,
        n_threads: int,
        out_path_raw: str,
        model_str: str,
        extra_obj: bool,
):
    if isinstance(n, int):
        rand_loop = range(n)
    else:
        rand_loop = n
    calls = []
    for i, dynamics_type in enumerate(dynamics_types):
        for i_rand in rand_loop:  # Should be 100
            calls.append(
                [
                    Date,
                    i_rand,
                    n_shooting,
                    dynamics_type,
                    ode_solver[i],
                    nstep,
                    n_threads,
                    out_path_raw,
                    model_str,
                    extra_obj,
                ]
            )
    return calls


# calls = generate_calls(
#     100, Date, n_shooting, dynamics_types, ode_solver, nstep, n_threads, out_path_raw, model_str, False,
# )

# pool_number = int(cpu_number / n_threads)
# with Pool(pool_number) as p:  # should be 4
#     p.map(miller_run.main, calls)

# Second call
# n_threads = 1  # Should be 8
# ode_solver = [OdeSolver.RK2, OdeSolver.RK2]
# dynamics_types = ["implicit", "root_implicit"]

# calls = generate_calls(
#     100, Date, n_shooting, dynamics_types, ode_solver, nstep, n_threads, out_path_raw, model_str, False,
# )
# pool_number = int(cpu_number / n_threads)
# with Pool(pool_number) as p:  # should be 4
#     p.map(miller_run.main, calls)

# calls = generate_calls(
#    100, Date, n_shooting, dynamics_types, ode_solver, nstep, n_threads, out_path_raw, model_str, True,
# )
# pool_number = int(cpu_number / n_threads)
# with Pool(pool_number) as p:  # should be 4
#    p.map(miller_run.main, calls)

# Third call missing ones
n_threads = 1  # Should be 8
ode_solver = [OdeSolver.RK2]
dynamics_types = ["root_implicit"]
irand_list = [31, 33, 35, 37, 43, 45, 55, 63, 65, 71]
calls = generate_calls(
    irand_list, Date, n_shooting, dynamics_types, ode_solver, nstep, n_threads, out_path_raw, model_str, False,
)
pool_number = int(cpu_number / n_threads)
with Pool(pool_number) as p:  # should be 4
    p.map(miller_run.main, calls)
