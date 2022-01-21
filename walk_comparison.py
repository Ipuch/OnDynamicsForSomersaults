import os, shutil
from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
from humanoid_2d import Humanoid2D
from humanoid_ocp import HumanoidOcp
from bioptim import Solver, OdeSolver


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


def delete_files(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def define_res_path(name: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_path = dir_path + "/" + name
    if out_path not in [x[0] for x in os.walk(dir_path)]:
        os.mkdir(out_path)
    else:
        print(f"deleting all files in {out_path}")
        delete_files(out_path)
    return out_path


def main():
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
