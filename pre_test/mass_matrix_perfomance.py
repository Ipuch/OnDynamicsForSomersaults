import biorbd_casadi as biorbd
from casadi import MX, SX, DM, Function, inv, solve, ldl_solve, mtimes
# MX : matrix symbolic
# SX : scalar symbolic
import numpy as np
import time

model_path = "../Model_JeCh_10DoFs.bioMod"
m = biorbd.Model(model_path)

q = MX.sym("q", m.nbQ(), 1)
M = m.massMatrix(q).to_mx()
M_func = Function("M_func", [q], [M], ["q"], ["M"]).expand()

M_inv = inv(M[:m.nbRoot(), :m.nbRoot()])
M_inv_func = Function("M", [q], [M_inv])

M_inv_solv = solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()))
M_inv_solv_func = Function("M_inv_solv", [q], [M_inv_solv])

M_inv_ldl = solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()), 'ldl')
M_inv_ldl_func = Function("M_inv_ldl", [q], [M_inv_ldl])

np.random.seed(0)
Q = np.random.random((m.nbQ(), 10000))

print("Inv")
tic = time.time()
for qi in Q.T:
    M_inv_func(qi)
toc = time.time() - tic
print(toc)

print("solv")
tic = time.time()
for qi in Q.T:
    M_inv_solv_func(qi)
toc = time.time() - tic
print(toc)

print("solve LDL")
tic = time.time()
for qi in Q.T:
    M_inv_ldl_func(qi)
toc = time.time() - tic
print(toc)

print("FD")
qdot = MX.sym("qdot", m.nbQ(), 1)
qddot_J = MX.sym("q", m.nbQ()-m.nbRoot(), 1)
tic = time.time()
for qi in Q.T:
    m.ForwardDynamics(qi, qi, qi)
toc = time.time() - tic
print(toc)

print("floating base dynamic")
qddot_J = MX.sym("q", m.nbQ()-m.nbRoot(), 1)
FD_fb = solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()), 'ldl') @ \
                 (-M[:m.nbRoot(), m.nbRoot():] @ qddot_J - m.NonLinearEffect(q, qdot).to_mx()[:m.nbRoot()])
FD_fb_func = Function("FD_fb", [q, qdot, qddot_J], [FD_fb])

tic = time.time()
for qi in Q.T:
    FD_fb_func(qi, qi, qi[m.nbRoot():])
toc = time.time() - tic
print(toc)

print("floating base dynamic")
qdot = MX.sym("qdot", m.nbQ(), 1)
qddot_J = MX.sym("q", m.nbQ()-m.nbRoot(), 1)
FD_fb = mtimes(solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()), 'ldl'),
                 (-M[:m.nbRoot(), m.nbRoot():] @ qddot_J - m.NonLinearEffect(q, qdot).to_mx()[:m.nbRoot()]))
FD_fb_func = Function("FD_fb", [q, qdot, qddot_J], [FD_fb])

tic = time.time()
for qi in Q.T:
    FD_fb_func(qi, qi, qi[m.nbRoot():])
toc = time.time() - tic
print(toc)