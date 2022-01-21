import numpy as np

from Comparison import Model, AcrobotOCP, Comparison
from bioptim import OdeSolver
import pickle

biobrd_model_path = "/home/puchaud/Projets_Python/My_bioptim_examples/Comparison/models/acrobot.bioMod"

ns_per_s = 100
t = 1
ns = int(100 * t)

X0 = np.zeros((4, ns + 1))
X0[0, :] = np.pi
X0[1, :] = np.linspace(0, np.pi, ns + 1)
nstep = 1
# ode_solver_list = [OdeSolver.RK4(n_integration_steps=nstep),
#                    OdeSolver.RK4(n_integration_steps=nstep * 2),
#                    OdeSolver.RK8(n_integration_steps=nstep),
#                    OdeSolver.CVODES(),
#                    OdeSolver.IRK(polynomial_degree=3, method='legendre'),
#                    OdeSolver.IRK(polynomial_degree=9, method='legendre'),
#                    OdeSolver.COLLOCATION(polynomial_degree=3, method='legendre'),
#                    OdeSolver.COLLOCATION(polynomial_degree=9, method='legendre')]
ode_solver_list = [OdeSolver.RK4(n_integration_steps=nstep), OdeSolver.RK4(n_integration_steps=nstep * 2)]

# for i in integrator_list:
#     A = AcrobotOCP(biobrd_model_path,
#                    tol=1e-4,
#                    online_optim=False,
#                    ode_solver=i,
#                    X0=X0,
#                    final_time=t)
#     A.solve()
#     print(i.rk_integrator.__name__)
#     print(A.sol.cost)
#     print(A.sol.real_time_to_optimize)

# A.sol.animate()
# A.sol.graphs()

comp = Comparison(
    ode_solver_list=ode_solver_list,
    tolerance_list=[1, 0.1, 0.01, 0.001, 1e-5, 1e-8],
    shooting_node_list=[100, 150, 200, 250],
    X0=X0,
    model_path=biobrd_model_path,
    ocp=AcrobotOCP,
)
comp.loop()
comp.graphs()
res = comp.df
f = open("df" + ".pckl", "wb")
pickle.dump(comp.df, f)
f.close()
print(res)
