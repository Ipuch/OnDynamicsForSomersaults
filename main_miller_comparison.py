
import os, shutil
from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
import numpy as np
from multiprocessing import Pool
from datetime import date
import smtplib, ssl

Date = date.today()
Date = Date.strftime("%d-%m-%y")
f = open(f"Historique_{Date}.txt", "w+")
f.write(" Debut ici \n\n\n")
f.close()

n_shooting = 150
duration = np.mean(np.array([1.44, 1.5, 1.545, 1.5, 1.545]))
dynamics_types = ["explicit", "root_explicit", "implicit", "root_implicit"]
nstep = 5
n_threads = 4 # Should be 8

calls = []
for weight in Weight_choices:
    for dynamics_type in dynamics_types:
        for i_rand in range(100):  # Should be 100
            calls.append([Date, i_rand, n_shooting, duration, dynamics_type, nstep, n_threads])

with Pool(1) as p: # should be 8
    p.map(miller_run.main, calls)

