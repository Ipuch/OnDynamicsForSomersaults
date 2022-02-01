import biorbd_casadi as biorbd
from casadi import MX, SX, DM, Function, inv, solve, ldl_solve, mtimes, lt, CodeGenerator, Importer, external

# MX : matrix symbolic
# SX : scalar symbolic
import numpy as np
import time

# unfinished

model_path = "../Model_JeCh_15DoFs.bioMod"
m = biorbd.Model(model_path)

q = MX.sym("q", m.nbQ(), 1)
qdot = MX.sym("qdot", m.nbQ(), 1)
qddot = MX.sym("qddot", m.nbQ(), 1)
qddot_J = MX.sym("qddot_J", m.nbQ() - m.nbRoot(), 1)
tau = MX.sym("tau", m.nbQ(), 1)
C = CodeGenerator("gen.c")

M = m.massMatrix(q).to_mx()
M_func = Function("M_func", [q], [M], ["q"], ["M"]).expand()
M_func_jacobian = M_func.jacobian()
M_func_hess = (M_func.jacobian()).jacobian()

C.add(M_func)
C.add(M_func.jacobian())
C.add((M_func.jacobian()).jacobian())

C.generate("gen")

# run this in terminal: gcc -fPIC -shared gengen.c -o gen.so
qrand = np.random.random(m.nbQ())

print("M")
M_func_ex = external("M_func", "./gen.so")
qrand = np.random.random(m.nbQ())
n = 10000
tic = time.time()
for i in range(n):
    M_func_ex(qrand)
toc = time.time() - tic
print(toc / n, " second")

tic = time.time()
for i in range(n):
    M_func(qrand)
toc = time.time() - tic
print(toc / n, " second")

print("M jacobian")
M_func_jacobian_ex = external("jac_M_func", "./gen.so")

n = 10000
tic = time.time()
for i in range(n):
    M_func_jacobian_ex(qrand, qrand)
toc = time.time() - tic
print(toc / n, " second")

tic = time.time()
for i in range(n):
    M_func_jacobian(qrand, qrand)
toc = time.time() - tic
print(toc / n, " second")

print("M hessian")
M_func_hessian_ex = external("jac_jac_M_func", "./gen.so")
n = 10000
tic = time.time()
for i in range(n):
    M_func_hessian_ex(
        qrand,
        qrand[:, np.newaxis] @ qrand[:, np.newaxis].T,
        np.repeat(qrand[:, np.newaxis] @ qrand[:, np.newaxis].T, 15, axis=0),
    )
toc = time.time() - tic
print(toc / n, " second")

tic = time.time()
for i in range(n):
    M_func_hess(
        qrand,
        qrand[:, np.newaxis] @ qrand[:, np.newaxis].T,
        np.repeat(qrand[:, np.newaxis] @ qrand[:, np.newaxis].T, 15, axis=0),
    )
toc = time.time() - tic
print(toc / n, " second")
