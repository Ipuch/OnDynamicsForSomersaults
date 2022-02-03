import biorbd  # biorbd_casadi as biorbd
import numpy as np
import time

m = biorbd.Model("../Model_JeCh_15DoFs.bioMod")

# print(f"\n\n\n --------------------- Random {i} -------------------------\n")
# Q_aleat = np.array([10, 20, 30, 44, 50, 60, 70, 80, 90, 100, 10, 9, 8, 7, 6]) / 100
Q_aleat = np.array([ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -2.8,
        0. ,  2.8,  0. ,  0. ])
# Q_aleat = np.zeros(15)
print(Q_aleat)
M = m.massMatrix(Q_aleat).to_array()
M_inv = np.linalg.inv(M)
M_inv2 = m.massMatrixInverse(Q_aleat).to_array()
print(np.round(M_inv - M_inv2, 8) == 0)

print(np.round(M_inv[:6, :6] - M_inv2[:6, :6], 8) == 0)

M_66_inv = np.linalg.inv(M[:6, :6])
print(np.round(M_66_inv - M_inv2[:6, :6], 8) == 0)

m.InverseDynamics(Q_aleat, np.zeros(15), np.zeros(15)).to_array()
