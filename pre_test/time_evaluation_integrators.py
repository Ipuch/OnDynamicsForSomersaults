import numpy as np
from bioptim import OdeSolver
from miller_ocp import MillerOcp
from time import perf_counter
import pandas as pd

ode_solvers = [OdeSolver.RK4(n_integration_steps=5),
               OdeSolver.RK4(n_integration_steps=5),
               OdeSolver.RK2(n_integration_steps=5),
               OdeSolver.RK2(n_integration_steps=5)]
dynamics_types = ["explicit", "root_explicit", "implicit", "root_implicit"]
times = np.zeros(4)

n_shooting = (125, 25)
duration = 1.545
n_threads = 8
model_path = "../Model_JeCh_15DoFs.bioMod"

np.random.seed(0)

n_eval = 1  #100000

for i in range(4):
    miller = MillerOcp(
        biorbd_model_path=model_path,
        duration=duration,
        n_shooting=n_shooting,
        ode_solver=ode_solvers[i],
        dynamics_type=dynamics_types[i],
        n_threads=n_threads,
        vertical_velocity_0=9.2,
        somersaults=4 * np.pi,
        twists=6 * np.pi,
    )
    nb_q = miller.biorbd_model[0].nbQ()
    nb_qdot = miller.biorbd_model[0].nbQdot()
    nb_qddot = miller.biorbd_model[0].nbQddot()
    nb_root = miller.biorbd_model[0].nbRoot()
    nb_tau = miller.biorbd_model[0].nbGeneralizedTorque()

    x = np.random.random(nb_q + nb_qdot)

    if dynamics_types[i] == "explicit":
        u = np.random.random(nb_tau)
        u = miller.mapping.options[0]["tau"].to_first.map(u)

    elif dynamics_types[i] == "root_explicit":
        u = np.random.random(nb_qddot - nb_root)

    elif dynamics_types[i] == "implicit":
        tau = np.random.random(nb_tau)
        tau_reduced = miller.mapping.options[0]["tau"].to_first.map(tau)
        nb_tau_reduced = len(tau_reduced)
        nb_qddot = miller.biorbd_model[0].nbQddot()
        u = np.random.random(nb_tau_reduced + nb_qddot)

    elif dynamics_types[i] == "root_implicit":
        u = np.random.random(nb_qddot)

    print(ode_solvers[i].__str__())
    print(dynamics_types[i].__str__())

    tic = perf_counter()
    for j in range(n_eval):
        xf, xall = miller.ocp.nlp[0].dynamics[0].function(x, u, [])
    toc = (perf_counter() - tic) / n_eval
    print("time : ", toc * 1e6, " microseconds")
    times[i] = toc * 1e6

percent = times / np.max(times) * 100
xfaster = np.divide(np.repeat(np.max(times), 4, axis=0), times)

np.savetxt('time_evaluation_integrators.out', (times,percent,xfaster), delimiter='    ', header="time, percent of time, x time faster (in microsec)")