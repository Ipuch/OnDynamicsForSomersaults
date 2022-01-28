import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver
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
        out_path_secondary_variables = args[9]
    else:
        Date = "24jan2022"
        i_rand = 0
        n_shooting = (125, 25)
        duration = 1.545
        dynamics_type = "explicit"
        ode_solver = OdeSolver.RK4
        nstep = 5
        n_threads = 1

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

    np.random.seed(i_rand)

    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1)
    solver.set_print_level(5)

    tic = time()
    sol = miller.ocp.solve(solver)
    toc = time() - tic

    if sol.status == 0:
        print(f"Time to solve dynamics_type={dynamics_type}, random={i_rand}: {toc}sec")

        f = open(f"{out_path_raw}/miller_{dynamics_type}_irand{i_rand}.pckl", "wb")
        data = {
            "computation_time": toc,
            "cost": sol.cost,
            "inf_du": sol.inf_du,
            "inf_pr": sol.inf_pr,
            "iterations": sol.iterations,
            "status": sol.status,
            "states": sol.states,
            "parameters": sol.parameters,
            "controls": sol.controls,
        }
        pickle.dump(data, f)
        f.close()


if __name__ == "__main__":
    main()
