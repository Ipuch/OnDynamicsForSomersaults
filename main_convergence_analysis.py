import os, shutil
from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
from bioptim import Solver, OdeSolver, MultiBodyDynamics
from miller_ocp import MillerOcp

# n_shooting = [(125, 25), (250, 50), (500, 100)]
n_shooting = [(50, 10), (55, 11), (60, 12)]
ode_solver = [OdeSolver.RK4()]
multibody_dynamics = [MultiBodyDynamics.IMPLICIT]


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


def main():
    model_path = "Model_JeCh_15DoFs.bioMod"
    out_path = "../OnDynamicsForSommersaults_results/raw_converge_analysis"

    comparison_parameters = ComparisonParameters(
        ode_solver=ode_solver,
        n_shooting=n_shooting,
        multibody_dynamics=multibody_dynamics,
        biorbd_model_path=model_path,
    )

    # --- Solve the program --- #
    solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solv.set_print_level(5)
    solv.set_maximum_iterations(1)

    comp = ComparisonAnalysis(Ocp=MillerOcp, Parameters=comparison_parameters, solver_options=solv)
    comp.loop()
    # comp.loop(res_path=out_path)

    f = open(f"{out_path}/df_{out_path}.pckl", "wb")
    pickle.dump(comp.df, f)
    f.close()

    f = open(f"{out_path}/comp_{out_path}.pckl", "wb")
    pickle.dump(comp, f)
    f.close()

    # comp.graphs(res_path=out_path, fixed_parameters={"multibody_dynamics": True}, show=True)
    comp.graphs(second_parameter="n_shooting", third_parameter="multibody_dynamics", res_path=out_path, show=True)
    # comp.graphs(second_parameter="n_shooting", third_parameter="multibody_dynamics", res_path=out_path, show=True)


if __name__ == "__main__":
    main()
