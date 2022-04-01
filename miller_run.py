import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver, Shooting
from miller_ocp import MillerOcp
import pickle
from time import time
from custom_dynamics.enums import MillerDynamics


def main(args: list = None):
    if args:
        Date = args[0]
        i_rand = args[1]
        n_shooting = args[2]
        dynamics_type = args[3]
        ode_solver = args[4]
        nstep = args[5]
        n_threads = args[6]
        out_path_raw = args[7]
        biorbd_model_path = args[8]
        extra_obj = args[9]
    else:
        Date = "11fev2022"
        i_rand = 0
        n_shooting = (125, 25)
        dynamics_type = "root_explicit"
        ode_solver = OdeSolver.RK4
        nstep = 5
        n_threads = 30
        out_path_raw = "../OnDynamicsForSommersaults_results/test"
        biorbd_model_path = "Model_JeCh_15DoFs.bioMod"
        extra_obj = True

    # to handle the random multi-start of the ocp
    np.random.seed(i_rand)
    # --- Solve the program --- #
    miller = MillerOcp(
        biorbd_model_path="Model_JeCh_15DoFs.bioMod",
        n_shooting=n_shooting,
        ode_solver=ode_solver(n_integration_steps=nstep),
        dynamics_type=dynamics_type,
        n_threads=n_threads,
        somersaults=4 * np.pi,
        twists=6 * np.pi,
        extra_obj=extra_obj,
    )
    filename = f"miller_{dynamics_type}_irand{i_rand}_extraobj{extra_obj}_{n_shooting[0]}_{n_shooting[1]}"
    outpath = f"{out_path_raw}/" + filename

    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(3000)
    solver.set_print_level(5)
    solver.set_linear_solver("ma57")

    print(f"##########################################################")
    print(
        f"Solving dynamics_type={dynamics_type}, i_rand={i_rand}," f"n_shooting={n_shooting}, extra_obj={extra_obj}\n"
    )
    print(f"##########################################################")

    tic = time()
    sol = miller.ocp.solve(solver)
    toc = time() - tic

    sol.print(cost_type=CostType.OBJECTIVES, to_console=False)

    print(f"##########################################################")
    print(
        f"Time to solve dynamics_type={dynamics_type}, i_rand={i_rand}, extra_obj={extra_obj}"
        f"n_shooting={n_shooting}, extra_obj={extra_obj}\n:\n {toc}sec\n"
    )
    print(f"##########################################################")

    sol_integrated = sol.integrate(
        shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, merge_phases=True, continuous=False
    )

    q_integrated = sol_integrated.states["q"]
    qdot_integrated = sol_integrated.states["qdot"]
    if (
        dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
        or dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT
    ):
        qddot_integrated = sol_integrated.states["qddot"]
    else:
        qddot_integrated = np.nan

    f = open(f"{outpath}.pckl", "wb")
    data = {
        "model_path": biorbd_model_path,
        "irand": i_rand,
        "extra_obj": extra_obj,
        "computation_time": toc,
        "cost": sol.cost,
        "detailed_cost": sol.detailed_cost,
        "iterations": sol.iterations,
        "status": sol.status,
        "states": sol.states,
        "controls": sol.controls,
        "parameters": sol.parameters,
        "dynamics_type": dynamics_type,
        "q_integrated": q_integrated,
        "qdot_integrated": qdot_integrated,
        "qddot_integrated": qddot_integrated,
        "n_shooting": n_shooting,
        "n_theads": n_threads,
    }
    pickle.dump(data, f)
    f.close()

    miller.ocp.save(sol, f"{outpath}.bo")


if __name__ == "__main__":
    main()
