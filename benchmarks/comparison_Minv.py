import biorbd  # biorbd_casadi as biorbd
import numpy as np
import time

m = biorbd.Model("../Model_JeCh_15DoFs.bioMod")

# print(f"\n\n\n --------------------- Random {i} -------------------------\n")
# Q_aleat = np.array([10, 20, 30, 44, 50, 60, 70, 80, 90, 100, 10, 9, 8, 7, 6]) / 100
Q_aleat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.8, 0.0, 2.8, 0.0, 0.0])
# Q_aleat = np.zeros(15)
print(Q_aleat)
print("\n")
tic = time.time()
for ii in range(1):
    M = m.massMatrix(Q_aleat).to_array()
    M_inv = np.linalg.inv(M)
toc = time.time() - tic
print(toc)
# print(M_inv)
# print("\n")
tic = time.time()
for ii in range(1):
    M_inv2 = m.massMatrixInverse(Q_aleat).to_array()
toc = time.time() - tic
print(toc)
# print(M_inv2)
# print("\n")
# print(np.round(np.triu(M_inv) - M_inv2, 9) == 0)
print(M)
print(np.round(M_inv - M_inv2, 8) == 0)
print(M_inv)
print(M_inv2)

print("base")
print(M_inv[:6, :6])
