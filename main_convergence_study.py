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
    calls = []
    for i, dynamics_type in enumerate(dynamics_types):
        for n_shooting in n_shooting_list:
            for i_rand in range(n):  # Should be 100
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


n_threads = 1
ode_solver = OdeSolver.RK2
dynamics_types = [MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT]

calls = generate_calls(
    repeat, Date, n_shooting_list_1, dynamics_types, ode_solver, nstep, n_threads, out_path_raw, model_str, True,
)
pool_number = 5
with Pool(pool_number) as p:
    p.map(miller_run.main, calls)


calls = generate_calls(
    repeat, Date, n_shooting_list_2, dynamics_types, ode_solver, nstep, n_threads, out_path_raw, model_str, True,
)
pool_number = 5
with Pool(pool_number) as p:
    p.map(miller_run.main, calls)


calls = generate_calls(
   repeat, Date, n_shooting_list_3, dynamics_types, ode_solver, nstep, n_threads, out_path_raw, model_str, True,
 )
pool_number = 5
with Pool(pool_number) as p:
    p.map(miller_run.main, calls)
