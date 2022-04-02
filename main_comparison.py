"""
This script runs 100 simulations per MillerDynamics such as MillerDynamics.EXPLICIT, MillerDynamics.ROOT_EXPLICIT,
MillerDynamics.ROOT_IMPLICIT, MillerDynamics.IMPLICIT, MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDOT, ...
It does it with  multiprocessing.Pool() to compare explicit and implicit formulations and equations of motion namely,
full-body dynamics and free-floating base dynamics.
This script was originally run on an AMD Ryzen 9 5950X processor and with 128 Go RAM.
"""
import os
from pathlib import Path
from typing import Union
import pickle
from multiprocessing import Pool, cpu_count
from datetime import date
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

cpu_number = cpu_count()
model_str = "Model_JeCh_15DoFs.bioMod"
n_shooting = (125, 25)
nstep = 5


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
) -> list:
    """
    Generate the list of calls to be used in multiprocessing

    Parameters
    ----------
    n : Union[int, list]
        The int of list of irand to be run
    Date : str
        The date of the run
    n_shooting : tuple
        The number of shooting points for each phase
    dynamics_types : list
        The list of dynamics types to be run such as MillerDynamics.EXPLICIT, MillerDynamics.IMPLICIT, MillerDynamics.EXPLICIT_IMPLICIT
    ode_solver : list
        The list of OdeSolver to be run such as OdeSolver.RK4, OdeSolver.RK2
    nstep : int
        The number of intermediate steps between two shooting points
    n_threads : int
        The number of threads to be used
    out_path_raw : str
        The path to store the raw results
    model_str : str
        The path to the bioMod model
    extra_obj : bool
        Whether to run with the extra objective or not (minimizing extra controls for implicit formulations)

    Returns
    -------
    calls: list
        The list of calls to be run
    """
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


def run_pool(calls: list, pool_nb: int):
    """
    Run the pool of processes

    Parameters
    ----------
    calls : list
        The list of calls to be run
    pool_nb : int
        The number of processes to be used in parallel
    """
    with Pool(pool_nb) as p:  # should be 4
        p.map(miller_run.main, calls)


def run_the_missing_ones(
    out_path_raw: str,
    Date,
    n_shooting: tuple,
    dynamics_types: list,
    ode_solver: list,
    nstep: int,
    n_threads: int,
    model_str: str,
    extra_obj: bool,
    pool_nb: int,
):
    """
    This function is used to run the process that were not run during the previous pool of processes

    Parameters
    ----------
    out_path_raw : str
        The path to store the raw results
    Date : str
        The date of the run
    n_shooting : tuple
        The number of shooting points for each phase
    dynamics_types : list
        The list of dynamics types to be run such as MillerDynamics.EXPLICIT, MillerDynamics.IMPLICIT, MillerDynamics.ROOT_EXPLICIT
    ode_solver : list
        The list of OdeSolver to be run such as OdeSolver.RK4, OdeSolver.RK2
    nstep : int
        The number of intermediate steps between two shooting points
    n_threads : int
        The number of threads to be used
    model_str : str
        The path to the bioMod model
    extra_obj : bool
        Whether to run with the extra objective or not (minimizing extra controls for implicit formulations)
    """
    # Run the one that did not run
    files = os.listdir(out_path_raw)
    files.sort()

    new_calls = {dynamics_types[0].value: [], dynamics_types[1].value: []}
    for i, file in enumerate(files):
        if file.endswith(".pckl"):
            p = Path(f"{out_path_raw}/{file}")
            file_path = open(p, "rb")
            data = pickle.load(file_path)
            if (
                data["dynamics_type"].value == dynamics_types[0].value
                or data["dynamics_type"].value == dynamics_types[1].value
            ):
                new_calls[data["dynamics_type"].value].append(data["irand"])

    list_100 = [i for i in range(0, 100)]

    dif_list = list(set(list_100) - set(new_calls[dynamics_types[0].value]))
    if dif_list:
        calls = generate_calls(
            dif_list,
            Date,
            n_shooting,
            [dynamics_types[1]],
            [ode_solver[1]],
            nstep,
            n_threads,
            out_path_raw,
            model_str,
            extra_obj,
        )
        run_pool(calls, pool_nb)

    dif_list = list(set(list_100) - set(new_calls[dynamics_types[1].value]))

    if dif_list:
        calls = generate_calls(
            dif_list,
            Date,
            n_shooting,
            [dynamics_types[1]],
            [ode_solver[1]],
            nstep,
            n_threads,
            out_path_raw,
            model_str,
            extra_obj,
        )
        run_pool(calls, pool_nb)


# Running explicit formulations full-body dynamics and free-floating dynamics
n_threads = 4
ode_solver = [OdeSolver.RK4, OdeSolver.RK4]
dynamics_types = [MillerDynamics.EXPLICIT, MillerDynamics.ROOT_EXPLICIT]

my_calls = generate_calls(
    100,
    Date,
    n_shooting,
    dynamics_types,
    ode_solver,
    nstep,
    n_threads,
    out_path_raw,
    model_str,
    False,
)
my_pool_number = int(cpu_number / n_threads)
run_pool(my_calls, my_pool_number)
run_the_missing_ones(
    out_path_raw, Date, n_shooting, dynamics_types, ode_solver, nstep, n_threads, model_str, False, my_pool_number
)

# Running implicit formulations full-body dynamics and free-floating dynamics
n_threads = 1
ode_solver = [OdeSolver.RK2, OdeSolver.RK2]
dynamics_types = [MillerDynamics.IMPLICIT, MillerDynamics.ROOT_IMPLICIT]

my_calls = generate_calls(
    100,
    Date,
    n_shooting,
    dynamics_types,
    ode_solver,
    nstep,
    n_threads,
    out_path_raw,
    model_str,
    True,
)

my_pool_number = int(cpu_number / n_threads)
run_pool(my_calls, my_pool_number)
run_the_missing_ones(
    out_path_raw, Date, n_shooting, dynamics_types, ode_solver, nstep, n_threads, model_str, True, my_pool_number
)

# Running implicit formulations with jerks as extra controls full-body dynamics and free-floating dynamics
n_threads = 1  # Should be 8
ode_solver = [OdeSolver.RK4, OdeSolver.RK4]
dynamics_types = [MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, MillerDynamics.ROOT_IMPLICIT_QDDDOT]

my_calls = generate_calls(
    100,
    Date,
    n_shooting,
    dynamics_types,
    ode_solver,
    nstep,
    n_threads,
    out_path_raw,
    model_str,
    True,
)

my_pool_number = int(cpu_number / n_threads)
run_pool(my_calls, my_pool_number)
run_the_missing_ones(
    out_path_raw, Date, n_shooting, dynamics_types, ode_solver, nstep, n_threads, model_str, True, my_pool_number
)
