import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import biorbd
import os
import pickle
import bioviz
from miller_ocp import MillerOcp
from bioptim import OdeSolver

file = open(f"/home/mickaelbegon/Documents/somersaults/OnDynamicsForSommersaults_results/test/miller_implicit_irand2.pckl", "rb")
data = pickle.load(file)

n_shooting = (125, 25)
ode_solver = OdeSolver.RK4(n_integration_steps=5)
duration = 1.545
n_threads = 8
model_path = "./Model_JeCh_15DoFs.bioMod"

# mettre une contrainte
# --- Solve the program --- #
miller = MillerOcp(
    biorbd_model_path=model_path,
    duration=duration,
    n_shooting=n_shooting,
    ode_solver=ode_solver,
    dynamics_type="root_explicit",
    n_threads=n_threads,
    vertical_velocity_0=9.2,
    somersaults=4 * np.pi,
    twists=6 * np.pi,
    use_sx=False,
)


#for ii in range(n_shooting[0]):
ii=0
x = data["states"][0]['all'][:, ii]
q = data["states"][0]['q'][:, ii]
qdot = data["states"][0]['qdot'][:, ii]

qddot = data["controls"][0]['qddot'][:, ii]
qddot_b = qddot[:6]
qddot_j = qddot[6:]
tau_j = data["controls"][0]['tau'][:, ii]


m = biorbd.Model("./Model_JeCh_15DoFs.bioMod")
MM = m.massMatrix(q).to_array()
NN = m.NonLinearEffect(q,qdot).to_array()
tau = m.InverseDynamics(q, qdot, qddot).to_array()
Minv = m.massMatrixInverse(q).to_array()

qdot0 = np.zeros(15)
NN0 = m.NonLinearEffect(q,qdot0).to_array()
tau0 = m.InverseDynamics(q, qdot0, qddot).to_array()
tau0_ = MM @ qddot + NN0

tau_0j = np.hstack((np.zeros(6), tau_j))

print("inverse/forward dynamics OK 1e-8")
tau_ = MM @ qddot + NN
print(tau[6:]-tau_j)
print(tau - tau_)
print(tau0 - tau0_)
print(qddot - m.ForwardDynamics(q, qdot, tau).to_array())
print(qddot - np.linalg.inv(MM) @ (tau_0j - NN))
print(qddot - Minv @ (tau_0j - NN))


param = data["parameters"]['all']
f = miller.ocp.nlp[0].dynamics[0].fun


print("inverse of masse matrix OK 1e-12")
M11inv = Minv[:6, :6]
M11inv_ = np.linalg.inv(m.massMatrix(q).to_array())[:6, :6]
M11invV = np.linalg.inv(m.massMatrix(q).to_array()[:6, :6])
print(M11inv - M11inv_)
print(M11inv - M11invV)


print("NLE OK 1e-9")
NLE_ = -m.massMatrix(q).to_array()[:6, 6:] @ qddot_j - m.NonLinearEffect(q, qdot).to_array()[:6]
NLE = -m.InverseDynamics(q, qdot, np.hstack((np.zeros((6)), qddot_j))).to_array()[:6]
print(NLE - NLE_)


print("Root explicit OK 1e-9")
f_qddot = f(x, qddot_j, param).toarray()[15:(15+6)].squeeze()
qddot_1 = M11inv_ @ NLE_
qddot_2 = M11inv @ NLE
qddot_3 = M11inv_ @ NLE
qddot_4 = M11invV @ NLE
print(f_qddot - qddot_1)
print(f_qddot - qddot_2)
print(qddot_1 - qddot_2)
print(qddot_3 - f_qddot)
print(MM[:6,:6] @ f_qddot + NLE)

print("AX=b, solve linalg : ca marche")
X = np.linalg.solve(MM[:6,:6], NLE)
print(f_qddot - X)
print(qddot_b - X)
print(MM[:6,:6] @ X  - NLE)


qddot_fj = np.hstack((X, qddot_j))
#qddot_fj = np.hstack((f_qddot, qddot_j))
tp = MM @ qddot_fj + m.NonLinearEffect(q,qdot).to_array()
print(tp - tau)






print( MM[6:,:6] @ qddot_b - (tau_j - MM[6:,6:] @ qddot_j - NN[6:] ))

#qddot_b2 = -np.linalg.inv(MM[6:,:6]) @ (tau_j - MM[6:,6:] @ qddot_j - NN[6:] )





- m.massMatrixInverse(q).to_array()[:6,:6] @ m.InverseDynamics(q, qdot, np.hstack((np.zeros((6)),qddot_j))).to_array()[:6]