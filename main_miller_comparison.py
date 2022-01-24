
import os, shutil
from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
import numpy as np
from multiprocessing import Pool
from datetime import date
import smtplib, ssl

# calls = []
# pwd = getpass.getpass()
# for a in range(-5, 6, 2):
#     for b in range(-5, 6, 2):
#         for c in range(-5, 6, 2):
#             for d in range(-5, 6, 2):
#                 calls.append([a / 10, b / 10, c / 10, d / 10, pwd])

n_threads = 4


Date = date.today()
Date = Date.strftime("%d-%m-%y")
f = open(f"Historique_{Date}.txt", "w+")
f.write(" Debut ici \n\n\n")
f.close()


calls = []
for weight in Weight_choices:
    for i_salto in range(len(Salto_1)):
        Salto1 = Salto_1[i_salto]
        Salto2 = Salto_2[i_salto]
        for i_rand in range(100):
            calls.append([Date, weight, Salto1, Salto2, i_rand, n_threads])

with Pool(2) as p:
    p.map(Trampo_Sauteur_multiStart.main, calls)

port = 465  # For SSL
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login("evidoux@gmail.com", "Josee9596")
    server.sendmail("evidoux@gmail.com", "evidoux@gmail.com", " * * * fini! * * * :D")






nstep = 1
ode_solver = [
    OdeSolver.RK4(n_integration_steps=nstep),
    OdeSolver.RK4(n_integration_steps=nstep * 2),
    OdeSolver.RK8(n_integration_steps=nstep),
    OdeSolver.CVODES(),
    OdeSolver.IRK(polynomial_degree=3, method="legendre"),
    OdeSolver.IRK(polynomial_degree=9, method="legendre"),
    OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre"),
    OdeSolver.COLLOCATION(polynomial_degree=9, method="legendre"),
]
# tolerance = [1, 1e-2, 1e-3, 1e-5, 1e-8]  # stay in scientific writting i.e. 1eX
# Est-ce qu'il sort avant?
# n_shooting = [5, 10, 20, 40]
# implicit_dynamics = [False]
# ode_solver = [
#     OdeSolver.RK4(n_integration_steps=nstep),
#     OdeSolver.COLLOCATION(polynomial_degree=9, method="legendre"),
# ]
tolerance = [
    # 1e-2,
    1e-6,
]  # stay in scientific writting i.e. 1eX
n_shooting = [10, 20]
implicit_dynamics = [False, True]


    model_path = Humanoid2D.HUMANOID_4DOF
    out_path = define_res_path(model_path.name)

    comparison_parameters = ComparisonParameters(
        ode_solver=ode_solver,
        tolerance=tolerance,
        n_shooting=n_shooting,
        implicit_dynamics=implicit_dynamics,
        biorbd_model_path=model_path.value,
    )

    # --- Solve the program --- #
    solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solv.set_print_level(5)

    comp = ComparisonAnalysis(Ocp=HumanoidOcp, Parameters=comparison_parameters, solver_options=solv)
    comp.loop(res_path=out_path)

    f = open(f"{out_path}/df_{model_path.name}.pckl", "wb")
    pickle.dump(comp.df, f)
    f.close()

    f = open(f"{out_path}/comp_{model_path.name}.pckl", "wb")
    pickle.dump(comp, f)
    f.close()

    # comp.graphs(res_path=out_path, fixed_parameters={"implicit_dynamics": True}, show=True)
    comp.graphs(second_parameter="n_shooting", third_parameter="implicit_dynamics", res_path=out_path, show=True)


if __name__ == "__main__":
    main()
