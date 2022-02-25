import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import biorbd
import os
import pickle
import bioviz
from miller_ocp import MillerOcp
from bioptim import OdeSolver

file = open(
    f"/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw/miller_root_implicit_irand13.pckl", "rb"
)
data = pickle.load(file)

n_shooting = (125, 25)
ode_solver = OdeSolver.RK4(n_integration_steps=5)
duration = 1.545
n_threads = 8
model_path = "../Model_JeCh_15DoFs.bioMod"
dynamics_type = "root_explicit"
# mettre une contrainte
# --- Solve the program --- #
miller = MillerOcp(
    biorbd_model_path=model_path,
    duration=duration,
    n_shooting=n_shooting,
    ode_solver=ode_solver,
    dynamics_type=dynamics_type,
    n_threads=n_threads,
    vertical_velocity_0=9.2,
    somersaults=4 * np.pi,
    twists=6 * np.pi,
    use_sx=False,
)

# for ii in range(n_shooting[0]):
ii = 0
x = data["states"][0]["all"][:, ii]
qddot_b = data["controls"][0]["all"][:6, ii]
qddot_j = data["controls"][0]["all"][6:, ii]
param = data["parameters"]["all"]
f = miller.ocp.nlp[0].dynamics[0].fun

xdot = f(x, qddot_j, param)
print(xdot[15:])
print(qddot_b)
# print(qddot_j)
# Explicit    vs   Implicit
print(xdot[15 : (15 + 6)] - qddot_b)

q = x[:15]
qdot = x[15:]
qddot = data["controls"][0]["all"][:, ii]
m = biorbd.Model("../Model_JeCh_15DoFs.bioMod")


tau = m.InverseDynamics(q, qdot, qddot).to_array()

qddot - m.ForwardDynamics(q, qdot, tau).to_array()
xdot[15:] - m.ForwardDynamics(q, qdot, tau).to_array()


m.massMatrixInverse(q).to_array()[:6, :6] - np.linalg.inv(m.massMatrix(q).to_array())[:6, :6]

m.InverseDynamics(q, qdot, np.hstack((np.zeros((6)), qddot_j))).to_array()[:6]
m.massMatrix(q).to_array()[:6, 6:] @ qddot_j + m.NonLinearEffect(q, qdot).to_array()[:6]


(
    xdot[15 : (15 + 6)]
    + m.massMatrixInverse(q).to_array()[:6, :6]
    @ m.InverseDynamics(q, qdot, np.hstack((np.zeros((6)), qddot_j))).to_array()[:6]
)


m.InverseDynamics(q, qdot, np.hstack((np.array(xdot[15 : (15 + 6)]).squeeze(), qddot_j))).to_array()[:6]
