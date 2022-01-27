import biorbd_casadi as biorbd
from casadi import MX, SX, DM, Function, inv, solve, ldl_solve, mtimes, lt
# MX : matrix symbolic
# SX : scalar symbolic
import numpy as np
import time

model_path = "../Model_JeCh_10DoFs.bioMod"
m = biorbd.Model(model_path)

q = MX.sym("q", m.nbQ(), 1)

M = m.massMatrix(q).to_mx()
M_func = Function("M_func", [q], [M], ["q"], ["M"]).expand()

Minv = m.massMatrixInverse(q).to_mx()
Minv_func = Function("Minv_func", [q], [Minv], ["q"], ["Minv"]).expand()
Minv_func_not_expand = Function("Minv_func", [q], [Minv], ["q"], ["Minv"])

M_inv = inv(M)
M_inv_func = Function("Minv", [q], [M_inv])

M_inv_solv = solve(M, MX.eye(m.nbQ()))
M_inv_solv_func = Function("M_inv_solv", [q], [M_inv_solv])

M_inv_root = inv(M[:m.nbRoot(), :m.nbRoot()])
M_inv_root_func = Function("M", [q], [M_inv_root])

M_inv_root_solv = solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()))
M_inv_root_solv_func = Function("M_inv_solv", [q], [M_inv_root_solv])

M_inv_root_ldl = solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()), 'ldl')
M_inv_root_ldl_func = Function("M_inv_ldl", [q], [M_inv_root_ldl])

M_inv_root_qr = solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()), 'qr')
M_inv_root_qr_func = Function("M_inv_qr", [q], [M_inv_root_qr])  # it still doesnt work with expand

np.random.seed(0)
Q = np.random.random((m.nbQ(), 10000))

print("Val")
qi = Q[:,0]
print(lt(M_inv_func(qi) - Minv_func(qi), 1e-9))
print(lt(M_inv_solv_func(qi) - Minv_func(qi), 1e-9))


print("Inv num")
tic = time.time()
for qi in Q.T:
    M_inv_func(qi)
toc = time.time() - tic
print(toc)

print("Inv analytic expand")
tic = time.time()
for qi in Q.T:
    Minv_func(qi)
toc = time.time() - tic
print(toc)

print("Inv analytic not expand")
tic = time.time()
for qi in Q.T:
    Minv_func_not_expand(qi)
toc = time.time() - tic
print(toc)

print("solv")
tic = time.time()
for qi in Q.T:
    M_inv_root_solv_func(qi)
toc = time.time() - tic
print(toc)

print("solve LDL")
tic = time.time()
for qi in Q.T:
    M_inv_root_ldl_func(qi)
toc = time.time() - tic
print(toc)

print("solve QR")
tic = time.time()
for qi in Q.T:
    M_inv_root_qr_func(qi)
toc = time.time() - tic
print(toc)

print("FD")
qdot = MX.sym("qdot", m.nbQ(), 1)
tau = MX.sym("tau", m.nbQ(), 1)
FD = m.ForwardDynamics(q, qdot, tau).to_mx()
FD_func = Function("FD", [q, qdot, tau], [FD]).expand()
tic = time.time()
for qi in Q.T:
    FD_func(qi, qi, qi)
toc = time.time() - tic
print(toc)

print("floating base dynamic with ldl")
qdot = MX.sym("qdot", m.nbQ(), 1)
qddot_J = MX.sym("q", m.nbQ() - m.nbRoot(), 1)
FD_fb = mtimes(solve(M[:m.nbRoot(), :m.nbRoot()], MX.eye(m.nbRoot()), 'ldl'),
               (-M[:m.nbRoot(), m.nbRoot():] @ qddot_J - m.NonLinearEffect(q, qdot).to_mx()[:m.nbRoot()]))
FD_fb_func = Function("FD_fb", [q, qdot, qddot_J], [FD_fb])

tic = time.time()
for qi in Q.T:
    FD_fb_func(qi, qi, qi[m.nbRoot():])
toc = time.time() - tic
print(toc)

print("floating base dynamic with analytic inv")
qdot = MX.sym("qdot", m.nbQ(), 1)
qddot_J = MX.sym("q", m.nbQ() - m.nbRoot(), 1)
FD_fb_inv = mtimes(Minv_func(q)[:m.nbRoot(), :m.nbRoot()],
                   (-M_func(q)[:m.nbRoot(), m.nbRoot():] @ qddot_J - m.NonLinearEffect(q, qdot).to_mx()[:m.nbRoot()]))
FD_fb_inv_func = Function("FD_fb_inv", [q, qdot, qddot_J], [FD_fb_inv]).expand()

tic = time.time()
for qi in Q.T:
    FD_fb_inv_func(qi, qi, qi[m.nbRoot():])
toc = time.time() - tic
print(toc)
