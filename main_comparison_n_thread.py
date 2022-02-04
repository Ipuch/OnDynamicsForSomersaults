import os, shutil

# from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
import numpy as np
from multiprocessing import Pool
from datetime import date
import smtplib, ssl
import miller_run
from bioptim import OdeSolver

out_path_raw = "../OnDynamicsForSommersaults_results/raw_test_threads"
out_path_secondary_variables = "../OnDynamicsForSommersaults_results/secondary_variables"

Date = date.today()
Date = Date.strftime("%d-%m-%y")

# duration = np.mean(np.array([1.44, 1.5, 1.545, 1.5, 1.545]))
n_shooting = (125, 25)
ode_solver = [OdeSolver.RK2, OdeSolver.RK2]
duration = 1.545
dynamics_types = ["implicit", "root_implicit"]
nstep = 5
n_threads = [32, 16, 8, 4]  # Should be 8

calls = []
for i, dynamics_type in enumerate(dynamics_types):
    for thread in n_threads:  # Should be 100
        calls.append(
            [
                Date,
                0,
                n_shooting,
                duration,
                dynamics_type,
                ode_solver[i],
                nstep,
                thread,
                out_path_raw,
                "Model_JeCh_15DoFs.bioMod",
            ]
        )
        print(calls)

with Pool(1) as p:  # should be 4
    p.map(miller_run.main, calls)

