"""
This script runs 30 simulations per implicit MillerDynamics such as MillerDynamics.ROOT_IMPLICIT,
MillerDynamics.IMPLICIT, MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDOT, ... from 150 to 840 shooting nodes to evaluate the
benefits of increasing the number of shooting nodes.
It does it with multiprocessing.Pool() (5 process in parallel) to compare implicit formulations
and equations of motion namely, full-body dynamics and free-floating base dynamics while increasing
the number of shooting nodes.
This script was originally run on an AMD Ryzen 9 5950X processor and with 128 Go RAM.
"""
import os
from multiprocessing import Pool, cpu_count
from datetime import date
import miller_run
from bioptim import OdeSolver
from custom_dynamics.enums import MillerDynamics

Date = date.today()
Date = Date.strftime("%d-%m-%y")

out_path_raw = "../OnDynamicsForSommersaults_results/raw_convergence" + Date
try:
    os.mkdir(out_path_raw)
except:
    print("../OnDynamicsForSommersaults_results/raw_convergence" + Date + " is already created ")

cpu_number = cpu_count()

n_shooting_list_1 = [(125, 25), (175, 35), (200, 40)]
n_shooting_list_2 = [(250, 50), (300, 60)]
n_shooting_list_3 = [(400, 80), (500, 100)]
n_shooting_list_4 = [(600, 120), (700, 140)]

model_str = "Model_JeCh_15DoFs.bioMod"
nstep = 5
repeat = 30
n_threads = 1
ode_solver = OdeSolver.RK2
dynamics_types = [MillerDynamics.IMPLICIT, MillerDynamics.ROOT_IMPLICIT]


# ode_solver = OdeSolver.RK4
# dynamics_types = [MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, MillerDynamics.ROOT_IMPLICIT_QDDDOT]


def generate_calls(
    n,
    Date,
    n_shooting_list: list,
    dynamics_types: list,
    ode_solver: OdeSolver,
    nstep: int,
    n_threads: int,
    out_path_raw: str,
    model_str: str,
    extra_obj: bool,
):
    """
    This functions generates the calls to the miller_run with different random seeds for each simulation
    while testing different dynamics and equations of motion.

    Parameters
    ----------
    n : int
        Number of random seeds to generate.
    Date : str
        Date of the simulation.
    n_shooting_list : list
        List of tuples with the number of shooting nodes to test.
    dynamics_types : list
        List of MillerDynamics to test.
    ode_solver : OdeSolver
        OdeSolver to use.
    nstep : int
        Number of integration steps.
    n_threads : int
        Number of threads to use.
    out_path_raw : str
        Path to store the raw results.
    model_str : str
        Path to the bioMod model.
    extra_obj : bool
        If True, the extra objective is used for implicit formulations.

    Returns
    -------
    calls : list
        List of calls to the miller_run function.
    """
    calls = []
    for i, dynamics_type in enumerate(dynamics_types):
        for n_shooting in n_shooting_list:
            for i_rand in range(n):
                calls.append(
                    [
                        Date,
                        i_rand,
                        n_shooting,
                        dynamics_type,
                        ode_solver,
                        nstep,
                        n_threads,
                        out_path_raw,
                        model_str,
                        extra_obj,
                    ]
                )
    return calls


calls = generate_calls(
    repeat,
    Date,
    n_shooting_list_1,
    dynamics_types,
    ode_solver,
    nstep,
    n_threads,
    out_path_raw,
    model_str,
    True,
)
pool_number = 5
with Pool(pool_number) as p:
    p.map(miller_run.main, calls)

calls = generate_calls(
    repeat,
    Date,
    n_shooting_list_2,
    dynamics_types,
    ode_solver,
    nstep,
    n_threads,
    out_path_raw,
    model_str,
    True,
)
pool_number = 5
with Pool(pool_number) as p:
    p.map(miller_run.main, calls)

calls = generate_calls(
    repeat,
    Date,
    n_shooting_list_3,
    dynamics_types,
    ode_solver,
    nstep,
    n_threads,
    out_path_raw,
    model_str,
    True,
)
pool_number = 5
with Pool(pool_number) as p:
    p.map(miller_run.main, calls)

calls = generate_calls(
    repeat,
    Date,
    n_shooting_list_4,
    dynamics_types,
    ode_solver,
    nstep,
    n_threads,
    out_path_raw,
    model_str,
    True,
)
pool_number = 5
with Pool(pool_number) as p:
    p.map(miller_run.main, calls)
