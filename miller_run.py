"""
This script runs the miller optimal control problem with a given set of parameters and save the results.
The main function is used in main_comparison.py and main_convergence.py. to run the different Miller optimal control problem.
"""
import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver, Shooting
from miller_ocp import MillerOcp
import pickle
from time import time
from custom_dynamics.enums import MillerDynamics
from IPython import embed


def main(args: list = None):
    """
    Main function for the miller_run.py script.
    It runs the optimization and saves the results of a Miller Optimal Control Problem.

    Parameters
    ----------
    args : list
        List of arguments containing the following:
        args[0] : date
            Date of the optimization.
        args[1] : i_rand
            Random seed.
        args[2] : n_shooting
            Number of shooting nodes.
        args[3] : dynamics_type
            Type of dynamics to use such as MillerDynamics.EXPLICIT, MillerDynamics.IMPLICIT, ...
        args[4] : ode_solver
            Type of ode solver to use such as OdeSolver.RK4, OdeSolver.RK2, ...
        args[5] : nstep
            Number of steps for the ode solver.
        args[6] : n_threads
            Number of threads to use.
        args[7] : out_path_raw
            Path to the raw results.
        args[8] : biorbd_model_path
            Path to the biorbd model.
        args[9] : extra_obj
            Extra objective to add to the cost function mainly for implicit formulations
    """
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
        dynamics_type = MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT
        ode_solver = OdeSolver.RK4
        nstep = 5
        n_threads = 3
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

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(10000)
    solver.set_print_level(5)
    solver.set_linear_solver("ma57")

    print(f"##########################################################")
    print(
        f"Solving dynamics_type={dynamics_type}, i_rand={i_rand}," f"n_shooting={n_shooting}, extra_obj={extra_obj}\n"
    )
    print(f"##########################################################")

    # --- time to solve --- #
    tic = time()
    sol = miller.ocp.solve(solver)
    toc = time() - tic

    states = sol.states[0]["all"]
    controls = sol.controls[0]["all"]
    parameters = sol.parameters["all"]

    states_2 = states[:, :2]
    for i in range(1, np.shape(states)[1]-1):
        states_2 = np.hstack((states_2, states[:, i:i+2]))

    vals = miller.ocp.nlp[0].J[3].weighted_function(states_2, [], [], 10, [], parameters[0]/125)
    np.sum(vals)

    # En Mayer avec vrai norme comme val : toujours -1.6674162217789566e-18

    q_modifs = np.zeros((15, 126))
    q_modifs[13:16] = states[13:16, :]




    sol.print(cost_type=CostType.OBJECTIVES, to_console=False)

    print(f"##########################################################")
    print(
        f"Time to solve dynamics_type={dynamics_type}, i_rand={i_rand}, extra_obj={extra_obj}"
        f"n_shooting={n_shooting}, extra_obj={extra_obj}\n:\n {toc}sec\n"
    )
    print(f"##########################################################")

    # --- Save the results --- #
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
