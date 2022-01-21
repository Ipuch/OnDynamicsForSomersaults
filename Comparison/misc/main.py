import matplotlib.pyplot as plt

import Pendulum2 as Pendulum
from bioptim import OdeSolver
import pickle
import numpy as np
import pandas as pd
from itertools import product
import biorbd_casadi as biorbd

model_path = "models/triple_pendulum.bioMod"
biorbd_model = biorbd.Model(model_path)
nQ = biorbd_model.nbQ()

# TODO: à ajouter OdeSolver.RK4(n_integration_steps=nstep*2
integrator_names = ["RK4", "RK8", "CVODES", "IRK", "COLLOCATION_legendre_3", "COLLOCATION_legendre_9"]

nstep = 1
integrator_list = [
    OdeSolver.RK4(n_integration_steps=nstep),
    OdeSolver.RK8(n_integration_steps=nstep),
    OdeSolver.CVODES(),
    OdeSolver.IRK(polynomial_degree=3, method="legendre"),
    OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre"),
    OdeSolver.COLLOCATION(polynomial_degree=9, method="legendre"),
]
ns_per_second = [50, 150, 250]
tol = [1e-2, 1e-5, 1e-8]

n_integrator = len(integrator_list)
n_node = len(ns_per_second)
n_tol = len(tol)

df = pd.DataFrame(
    list(product(integrator_names, ns_per_second, tol)),
    columns=["integrator", "node per second", "optimizer tolerance"],
)

sol_list = [[[[] for i in range(n_tol)] for i in range(n_node)] for i in range(n_integrator)]
t = np.zeros((n_integrator * n_node * n_tol, 1))
iterations = np.zeros((n_integrator * n_node * n_tol, 1))
f_obj = np.zeros((n_integrator * n_node * n_tol, 1))
dynamic_consistency = np.zeros((n_integrator * n_node * n_tol, 5))
constraints = np.zeros((n_integrator * n_node * n_tol, 1))
cpt = 0
constraints = np.zeros((n_integrator * n_node * n_tol, 1))
states_list = n_node * [[]]
states_rk45 = n_node * [[]]
controls_list = n_node * [[]]
time_vector = n_node * [[]]
for i_ns in range(n_node):
    states_list[i_ns] = np.zeros((nQ * 2, ns_per_second[i_ns] + 1, n_integrator, n_tol))
    states_rk45[i_ns] = np.zeros((nQ * 2, ns_per_second[i_ns] + 1, n_integrator, n_tol))
    controls_list[i_ns] = np.zeros((nQ, ns_per_second[i_ns] + 1, n_integrator, n_tol))
    time_vector[i_ns] = np.zeros((1, ns_per_second[i_ns] + 1, n_integrator, n_tol))

# Solving for every conditions
for i_int in range(n_integrator):
    for i_ns in range(n_node):
        for i_tol in range(n_tol):
            print("Solve with")
            print("########## Integrator ##########")
            print(integrator_list[i_int])
            print("########## node / second ##########")
            print(ns_per_second[i_ns])
            print("########## Tolerance on IPOPT ##########")
            print(tol[i_tol])

            ocp, sol = Pendulum.main(
                ode_solver=integrator_list[i_int], n_shooting_per_second=ns_per_second[i_ns], tol=tol[i_tol]
            )
            # filling dataframe
            t[cpt, 0] = sol.real_time_to_optimize
            iterations[cpt, 0] = sol.iterations
            f_obj[cpt, 0] = sol.cost
            dynamic_consistency[cpt, :] = Pendulum.compute_error_single_shooting(ocp, sol, 1)
            constraints[cpt, 0] = np.sqrt(np.mean(sol.constraints.toarray() ** 2))

            states_rk45[i_ns][:, :, i_int, i_tol] = Pendulum.integrate_sol(ocp, sol)

            # filling states, controls and time
            if ocp.nlp[0].ode_solver.is_direct_collocation:
                n = ocp.nlp[0].ode_solver.polynomial_degree + 1
                states_list[i_ns][:, :, i_int, i_tol] = sol.states["all"][:, ::n]
            else:
                states_list[i_ns][:, :, i_int, i_tol] = sol.states["all"]

            controls_list[i_ns][:, :, i_int, i_tol] = sol.controls["all"]
            time_vector[i_ns][:, :, i_int, i_tol] = np.linspace(0, sol.phase_time[0 + 1], sol.ns[0] + 1)

            print("Done with")
            print("########## Integrator ##########")
            print(integrator_list[i_int])
            print("########## node / second ##########")
            print(ns_per_second[i_ns])
            print("########## Tolerance on IPOPT ##########")
            print(tol[i_tol])
            cpt = cpt + 1

# saving optimal control outputs
d = {
    "time": time_vector,
    "states": states_list,
    "states_rk45": states_rk45,
    "controls": controls_list,
    "n_integrator": n_integrator,
    "n_node": n_node,
    "n_tol": n_tol,
    "integrator_names": integrator_names,
    "ns_per_second": ns_per_second,
    "tol": tol,
}

f = open("d" + ".pckl", "wb")
pickle.dump(d, f)
f.close()

df["time"] = t
df["iter"] = iterations
df["objective function value"] = f_obj
df["translation dynamic consistency"] = dynamic_consistency[:, 0]
df["rotation dynamic consistency"] = dynamic_consistency[:, 1]
df["linear velocity dynamic consistency"] = dynamic_consistency[:, 2]
df["angular velocity dynamic consistency"] = dynamic_consistency[:, 3]
df["dynamic consistency"] = dynamic_consistency[:, 4]
df["constraints"] = constraints[:, 0]

f = open("df2" + ".pckl", "wb")
pickle.dump(df, f)
f.close()
