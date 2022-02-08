import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver, Shooting
from miller_ocp import MillerOcp
import pickle
from time import time


def main(args=None):
    if args:
        Date = args[0]
        i_rand = args[1]
        n_shooting = args[2]
        duration = args[3]
        dynamics_type = args[4]
        ode_solver = args[5]
        nstep = args[6]
        n_threads = args[7]
        out_path_raw = args[8]
        biorbd_model_path = args[9]
    else:
        Date = "3fev2022"
        i_rand = 0
        n_shooting = (125, 25)
        duration = 1.545
        dynamics_type = "root_explicit"
        ode_solver = OdeSolver.RK4
        nstep = 5
        n_threads = 30
        out_path_raw = "../OnDynamicsForSommersaults_results/test"
        biorbd_model_path = "Model_JeCh_15DoFs.bioMod"

    # to handle the random multi-start of the ocp
    np.random.seed(i_rand)
    # --- Solve the program --- #
    miller = MillerOcp(
        biorbd_model_path="Model_JeCh_15DoFs.bioMod",
        duration=duration,
        n_shooting=n_shooting,
        ode_solver=ode_solver(n_integration_steps=nstep),
        dynamics_type=dynamics_type,
        n_threads=n_threads,
        vertical_velocity_0=9.2,
        somersaults=4 * np.pi,
        twists=6 * np.pi,
    )

    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1500)
    solver.set_print_level(5)
    solver.set_linear_solver("ma57")

    print(f"##########################################################")
    print(f"Solving dynamics_type={dynamics_type}, i_rand={i_rand}\n")
    print(f"##########################################################")

    tic = time()
    sol = miller.ocp.solve(solver)
    toc = time() - tic

    sol.print(to_console=False)

    # if sol.status == 0:
    print(f"##########################################################")
    print(f"Time to solve dynamics_type={dynamics_type}, i_rand={i_rand}: {toc}sec\n")
    print(f"##########################################################")

    sol_integrated = sol.integrate(
        shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, merge_phases=True, continuous=False
    )

    q_integrated = sol_integrated.states["q"]
    qdot_integrated = sol_integrated.states["qdot"]

    f = open(f"{out_path_raw}/miller_{dynamics_type}_i_rand{i_rand}.pckl", "wb")
    data = {
        "model_path": biorbd_model_path,
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
        "n_shooting": n_shooting,
        "n_theads": n_threads,
    }
    pickle.dump(data, f)
    f.close()

    miller.ocp.save(sol, f"{out_path_raw}/miller_{dynamics_type}_irand{i_rand}.bo")


if __name__ == "__main__":
    main()
