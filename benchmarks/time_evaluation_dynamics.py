"""
This script performs the time evaluation of all the MillerDynamics without considering the optimal control problem.
"""
import biorbd_casadi as biorbd
from casadi import MX, Function, solve, mtimes, vertcat
import numpy as np
import time

model_path = "../Model_JeCh_15DoFs.bioMod"
m = biorbd.Model(model_path)

q = MX.sym("q", m.nbQ(), 1)

M = m.massMatrix(q).to_mx()
M_func = Function("M_func", [q], [M], ["q"], ["M"]).expand()
M66_func = Function("M_func", [q], [M[:6, :6]], ["q"], ["M66"]).expand()

Minv = m.massMatrixInverse(q).to_mx()
Minv_func = Function("Minv_func", [q], [Minv], ["q"], ["Minv"]).expand()

np.random.seed(0)
n = 10000
Q = np.random.random((m.nbQ(), n))
t = np.zeros(4)

print("FD")
qdot = MX.sym("qdot", m.nbQ(), 1)
tau = MX.sym("tau", m.nbQ(), 1)
FD = m.ForwardDynamics(q, qdot, tau).to_mx()
FD_func = Function("FD", [q, qdot, tau], [FD]).expand()
tic = time.time()
for qi in Q.T:
    FD_func(qi, qi, qi)
toc = time.time() - tic
print(toc / n, " second")
t[0] = toc / n
#
print("Inverse Dynamics")
qddot = MX.sym("qddot", m.nbQ(), 1)
ID = m.InverseDynamics(q, qdot, qddot).to_mx()
ID_func = Function("ID", [q, qdot, qddot], [ID]).expand()
tic = time.time()
for qi in Q.T:
    ID_func(qi, qi, qi)
toc = time.time() - tic
print(toc / n, " second")
t[1] = toc / n

print("False, need to be recomputed with inv(M[:6,:6]), floating base forward dynamics with analytic inv V2")
#
# qdot = MX.sym("qdot", m.nbQ(), 1)
qddot_J = MX.sym("qddot_J", m.nbQ() - m.nbRoot(), 1)
FD_fb_v2 = mtimes(Minv_func(q)[: m.nbRoot(), : m.nbRoot()], -ID_func(q, qdot, vertcat(MX.zeros((6, 1)), qddot_J))[:6])
FD_fb_v2_func = Function("FD_fb_v2", [q, qdot, qddot_J], [FD_fb_v2]).expand()

tic = time.time()
for qi in Q.T:
    FD_fb_v2_func(qi, qi, qi[m.nbRoot() :])
toc = time.time() - tic
print(toc / n, " second")
t[2] = toc / n

print("floating base forward dynamics with solve ldl inv, NLE et M66 casadi_func expand")
# qddot_J = MX.sym("qddot_J", m.nbQ() - m.nbRoot(), 1)
NLE = ID_func(q, qdot, vertcat(MX.zeros((6, 1)), qddot_J))[:6]
NLE_func = Function("NLE", [q, qdot, qddot_J], [NLE]).expand()
FD_fb_v2 = solve(M66_func(q), -NLE_func(q, qdot, qddot_J), "ldl")
FD_fb_v2_func = Function("FD_fb_v2", [q, qdot, qddot_J], [FD_fb_v2])

tic = time.time()
for qi in Q.T:
    FD_fb_v2_func(qi, qi, qi[m.nbRoot() :])
toc = time.time() - tic
print(toc / n, " second")
t[2] = toc / n

print("floating base forward dynamics with solve ldl inv - no expand")
ID_func = Function("ID", [q, qdot, qddot], [ID])
NLE = ID_func(q, qdot, vertcat(MX.zeros((6, 1)), qddot_J))[:6]
NLE_func = Function("NLE", [q, qdot, qddot_J], [NLE])
M66_func = Function("M_func", [q], [M[:6, :6]], ["q"], ["M66"])
FD_fb_v3 = solve(M66_func(q), -NLE_func(q, qdot, qddot_J), "ldl")
FD_fb_v3_func = Function("FD_fb_v3", [q, qdot, qddot_J], [FD_fb_v3])

tic = time.time()
for qi in Q.T:
    FD_fb_v3_func(qi, qi, qi[m.nbRoot() :])
toc = time.time() - tic
print(toc / n, " second")
t[2] = toc / n

print("floating base inverse dynamics")
ID = m.InverseDynamics(q, qdot, qddot).to_mx()[:6]
ID_fb_func = Function("ID", [q, qdot, qddot], [ID]).expand()
tic = time.time()
for qi in Q.T:
    ID_fb_func(qi, qi, qi)
toc = time.time() - tic
print(toc / n, " second")
t[3] = toc / n

np.savetxt(
    "time_evaluation_dynamics.out",
    t * 1e6,
    delimiter="    ",
    header="FD, ID, floating base FD, floating base ID in microsec",
)
