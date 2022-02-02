import biorbd as biorbd
from casadi import MX, SX, DM, Function, inv, solve, ldl_solve, mtimes, lt

# MX : matrix symbolic
# SX : scalar symbolic
import numpy as np
from time import perf_counter

model_path = "../Model_JeCh_15DoFs.bioMod"
m = biorbd.Model(model_path)

q = np.random.random(m.nbQ()) * 10
qdot = np.random.random(m.nbQ()) * 10
qddot = np.random.random(m.nbQ()) * 100
qddot[:6] = 0

N = m.NonLinearEffect(q, qdot).to_array()
M = m.massMatrix(q).to_array()

print(M[:6, 6:] @ qddot[6:] + N[:6])
print(m.InverseDynamics(q, qdot, qddot).to_array()[:6])

print(M[:6, 6:] @ qddot[6:] + N[:6] - m.InverseDynamics(q, qdot, qddot).to_array()[:6])