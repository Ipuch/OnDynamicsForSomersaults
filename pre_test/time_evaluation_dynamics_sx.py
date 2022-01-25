import biorbd_casadi as biorbd
from casadi import MX, SX, DM, Function, inv, solve, ldl_solve, mtimes, lt
# MX : matrix symbolic
# SX : scalar symbolic
import numpy as np
from time import perf_counter

model_path = "../Model_JeCh_15DoFs.bioMod"
m = biorbd.Model(model_path)

q = MX.sym("q", m.nbQ(), 1)
qdot = MX.sym("qdot", m.nbQ(), 1)
qddot = MX.sym("qddot", m.nbQ(), 1)
qddot_J = MX.sym("qddot_J", m.nbQ() - m.nbRoot(), 1)
tau = MX.sym("tau", m.nbQ(), 1)

M = m.massMatrix(q).to_mx()
M_func = Function("M_func", [q], [M], ["q"], ["M"]).expand()

Minv = m.massMatrixInverse(q).to_mx()
Minv_func = Function("Minv_func", [q], [Minv], ["q"], ["Minv"])

N = m.NonLinearEffect(q, qdot).to_mx()
N_func = Function("N_func", [q, qdot], [N], ["q", "qdot"], ["N"]).expand()

FD = m.ForwardDynamics(q, qdot, tau).to_mx()
FD_func = Function("FD", [q, qdot, tau], [FD]).expand()

ID = m.InverseDynamics(q, qdot, qddot).to_mx()
ID_func = Function("ID", [q, qdot, qddot], [ID]).expand()

q_sx = SX.sym("q", m.nbQ(), 1)
qdot_sx = SX.sym("qdot", m.nbQ(), 1)
qddot_sx = SX.sym("qddot", m.nbQ(), 1)
qddot_J_sx = SX.sym("qddot_J", m.nbQ() - m.nbRoot(), 1)
tau_sx = SX.sym("tau", m.nbQ(), 1)

FD_fb = mtimes(Minv_func(q_sx)[:m.nbRoot(), :m.nbRoot()], -M_func(q_sx)[:m.nbRoot(), m.nbRoot():] @ qddot_J_sx
               - N_func(q_sx, qdot_sx)[:m.nbRoot()])
FD_fb_func = Function("FD_fb", [q_sx, qdot_sx, qddot_J_sx], [FD_fb]).expand()

ID_fb = M_func(q_sx)[:m.nbRoot(), :] @ qddot_sx + N_func(q_sx, qdot_sx)[:m.nbRoot()]
ID_fb_func = Function("ID_fb", [q_sx, qdot_sx, qddot_sx], [ID_fb]).expand()

ID_full = M_func(q_sx) @ qddot_sx + N_func(q_sx, qdot_sx)
ID_full_func = Function("ID_fb", [q_sx, qdot_sx, qddot_sx], [ID_full]).expand()

ID6 = m.InverseDynamics(q, qdot, qddot).to_mx()[:m.nbRoot()] #MX
ID6_func = Function("ID_6", [q, qdot, qddot], [ID6]).expand() #SX

nn = 100000
print("Minv")
Minv_func = Function("Minv_func", [q], [Minv], ["q"], ["Minv"]).expand()
tic = perf_counter()
for i in range(nn):
    Minv_func(qi)
toc = perf_counter() - tic
print(toc/nn, " second")

print("Minv 6")
Minv6_func = Function("Minv_func", [q], [Minv_func(q)[:6, :6]], ["q"], ["Minv6"]).expand()
tic = perf_counter()
for i in range(nn):
    Minv6_func(qi)
toc = perf_counter() - tic
print(toc/nn, " second")

print("N full")
tic = perf_counter()
for i in range(nn):
    N_func(qi, qi)
toc = perf_counter() - tic
print(toc/nn, " second")

print("N [:6]")
N6_func = Function("N6_func", [q_sx, qdot_sx], [N_func(q_sx, qdot_sx)[:6]]).expand()
tic = perf_counter()
for i in range(nn):
    N6_func(qi, qi)
toc = perf_counter() - tic
print(toc/nn, " second")


print("misc full [:6]")
misc_func = Function("N6_func", [q_sx, qdot_sx], [ Minv6_func(q_sx) * N_func(q_sx, qdot_sx)[:6] ]).expand()
tic = perf_counter()
for i in range(nn):
    misc_func(qi, qi)
toc = perf_counter() - tic
print(toc/nn, " second")

print("misc full")
misc_func = Function("N6_func", [q_sx, qdot_sx], [ (Minv_func(q_sx) * N_func(q_sx, qdot_sx))]).expand()
tic = perf_counter()
for i in range(nn):
    misc_func(qi, qi)
toc = perf_counter() - tic
print(toc/nn, " second")

print("forward dynamics aba")
tic = perf_counter()
for i in range(nn):
    FD_func(qi, qi, qi)
toc = perf_counter() - tic
print(toc/nn, " second")

n = 100000
Q = np.random.random((m.nbQ(), n))

print("FD")
tic = perf_counter()
for qi in Q.T:
    FD_func(qi, qi, qi)
toc = perf_counter() - tic
print(toc/n, " second")

print("Inverse Dynamics")
tic = perf_counter()
for qi in Q.T:
    ID_func(qi, qi, qi)
toc = perf_counter() - tic
print(toc/n, " second")

print("floating base forward dynamics with analytic inv")
q_SX = SX.sym("q_SX", m.nbQ(), 1)
tic = perf_counter()
for qi in Q.T:
    FD_fb_func(qi, qi, qi[m.nbRoot():])
toc = perf_counter() - tic
print(toc/n, " second")

print("floating base inverse dynamics")
tic = perf_counter()
for qi in Q.T:
    ID_fb_func(qi, qi, qi)
toc = perf_counter() - tic
print(toc/n, " second")

print("Inverse Dynamics with first 6 dofs")
tic = perf_counter()
for qi in Q.T:
    ID6_func(qi, qi, qi)
toc = perf_counter() - tic
print(toc/n, " second")

