import os, shutil

# from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
import numpy as np
from multiprocessing import Pool
from datetime import date
import smtplib, ssl
import miller_run
from bioptim import OdeSolver

out_path_raw = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw"
out_path_secondary_variables = (
    "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/secondary_variables"
)

Date = date.today()
Date = Date.strftime("%d-%m-%y")
f = open(f"Historique_{Date}.txt", "w+")
f.write(" Debut ici \n\n\n")
f.close()

# duration = np.mean(np.array([1.44, 1.5, 1.545, 1.5, 1.545]))
n_shooting = (125, 25)
ode_solver = [OdeSolver.RK4, OdeSolver.RK4, OdeSolver.RK2, OdeSolver.RK2]
duration = 1.545
dynamics_types = ["explicit", "root_explicit", "implicit", "root_implicit"]
nstep = 5
n_threads = 1  # Should be 8

calls = []
for i, dynamics_type in enumerate(dynamics_types):
    for i_rand in range(100):  # Should be 100
        calls.append(
            [
                Date,
                i_rand,
                n_shooting,
                duration,
                dynamics_type,
                ode_solver[i],
                nstep,
                n_threads,
                out_path_raw,
                out_path_secondary_variables,
            ]
        )

with Pool(4) as p:  # should be 8
    p.map(miller_run.main, calls)
