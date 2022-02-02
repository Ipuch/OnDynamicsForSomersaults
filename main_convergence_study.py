import os, shutil
from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
from bioptim import Solver, OdeSolver, CostType, Shooting
from miller_ocp import MillerOcp
import numpy as np
from time import time

# n_shooting = [(125, 25), (250, 50), (500, 100)]
n_shooting_list = [(50, 10), (55, 11), (60, 12)]
ode_solver = [OdeSolver.RK2()]


def main():
    model_path = "Model_JeCh_15DoFs.bioMod"
    out_path = "../OnDynamicsForSommersaults_results/raw_converge_analysis"

    ode_solver = OdeSolver.RK4(n_integration_steps=5)
    duration = 1.545
    n_threads = 8
    model_path = "Model_JeCh_15DoFs.bioMod"
    dynamics_type = "implicit"  # "root_implicit"
    np.random.seed(0)

    for n_shooting in n_shooting_list:

        # --- Solve the program --- #
        miller = MillerOcp(
            biorbd_model_path=model_path,
            duration=duration,
            n_shooting=n_shooting,
            ode_solver=ode_solver,
            dynamics_type=dynamics_type,
            n_threads=n_threads,
            vertical_velocity_0=9.2,
            somersaults=4 * np.pi,
            twists=6 * np.pi,
            use_sx=False,
        )

        miller.ocp.print(to_console=True)
        miller.ocp.add_plot_penalty(CostType.ALL)

        solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
        solver.set_maximum_iterations(1000)
        solver.set_print_level(5)
        solver.set_linear_solver("ma57")
        # solver.set_linear_solver("mumps")

        print(f"##########################################################")
        print(f"Solving dynamics_type={dynamics_type}, n_shooting ={n_shooting}\n")
        print(f"##########################################################")

        tic = time()
        sol = miller.ocp.solve(solver)
        toc = time() - tic

        # if sol.status == 0:
        print(f"##########################################################")
        print(f"Time to solve dynamics_type={dynamics_type}, n_shooting={n_shooting}: {toc}sec\n")
        print(f"##########################################################")

        sol_integrated = sol.integrate(
            shooting_type=Shooting.SINGLE, keep_intermediate_points=True, merge_phases=True, continuous=False
        )

        q_integrated = sol_integrated.states["q"]
        qdot_integrated = sol_integrated.states["qdot"]

        f = open(f"{out_path}/miller_{dynamics_type}_{n_shooting[0]}_{n_shooting[1]}.pckl", "wb")
        data = {
            "model_path": model_path,
            "computation_time": toc,
            "cost": sol.cost,
            # "inf_du" : sol.inf_du,
            "iterations": sol.iterations,
            # "inf_pr" : sol.inf_pr,
            "status": sol.status,
            "states": sol.states,
            "controls": sol.controls,
            "parameters": sol.parameters,
            "dynamics_type": dynamics_type,
            "q_integrated": q_integrated,
            "qdot_integrated": qdot_integrated,
        }
        pickle.dump(data, f)
        f.close()


if __name__ == "__main__":
    main()
