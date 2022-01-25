import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver
from miller_ocp import MillerOcp
import time
import os
import shutil
from Comparison import ComparisonAnalysis, ComparisonParameters

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


def main(args=None):
    if args:
        Date = args[0]
        i_rand = args[1]
        n_shooting = args[2] # 150
        duration = args[3] # 1.545
        dynamics_type = args[4] # "root_implicit"  # "explicit"  # "implicit"  # "root_explicit"  # "root_implicit"

    # --- Solve the program --- #
    miller = MillerOcp(
        biorbd_model_path="Model_JeCh_10DoFs.bioMod",
        duration=duration,
        n_shooting=n_shooting,
        ode_solver=OdeSolver.RK4(n_integration_steps=5),
        dynamics_type=dynamics_type,
        n_threads=8,
        vertical_velocity_0=9.2,
        somersaults=4 * np.pi,
        twists=6 * np.pi,
    )


    # miller.ocp.add_plot_penalty(CostType.ALL)

    np.random.seed(i_rand)

    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1000)
    solver.set_print_level(5)
    solver.set_print_level(5)

    tic = time()
    sol = miller.ocp.solve(solver)
    toc = time() - tic
    print(f"Time to solve dynamics_type={dynamics_type}, random={i_rand}: {toc}sec")


    comp = ComparisonAnalysis(Ocp=HumanoidOcp, Parameters=comparison_parameters, solver_options=solv)
    comp.loop(res_path=out_path)

    f = open(f"{out_path}/df_{model_path.name}.pckl", "wb")
    pickle.dump(comp.df, f)
    f.close()

    f = open(f"{out_path}/comp_{model_path.name}.pckl", "wb")
    pickle.dump(comp, f)
    f.close()

    # --- Show results --- #
    # print(sol.status)
    # sol.print()
    # sol.graphs()
    # sol.animate()
    # sol.animate(nb_frames=-1, show_meshes=False) # show_mesh=True

if __name__ == "__main__":
    main()

