"""
This script demonstrate how to compute free floating base dynamics right term with inverse dynamics algorithm by setting
the free floating base acceleration to zero using biorbd library.
"""
import biorbd as biorbd
import numpy as np

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
