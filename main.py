import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver
from miller_ocp import MillerOcp
from miller_viz import add_custom_plots
from datetime import datetime
from custom_dynamics.enums import MillerDynamics


def main(dynamics_type, thread, solver, extra_obj: bool = False):
    n_shooting = (125, 25)
    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # n_threads = 8
    model_path = "Model_JeCh_15DoFs.bioMod"
    # dynamics_type = "root_explicit"  # "implicit"  # "explicit"  # "root_explicit"  # "root_implicit"
    # mettre une contrainte
    # --- Solve the program --- #
    miller = MillerOcp(
        biorbd_model_path=model_path,
        n_shooting=n_shooting,
        ode_solver=solver,
        dynamics_type=dynamics_type,
        n_threads=thread,
        somersaults=4 * np.pi,
        twists=6 * np.pi,
        use_sx=False,
        extra_obj=extra_obj,
    )

    # miller.ocp.print(to_console=True)

    add_custom_plots(miller.ocp, dynamics_type)
    miller.ocp.add_plot_penalty(CostType.CONSTRAINTS)
    np.random.seed(203)

    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(10000)
    solver.set_print_level(5)
    solver.set_linear_solver("ma57")
    # solver.set_limited_memory_max_history(500)

    sol = miller.ocp.solve(solver)

    # --- Show results --- #
    # if sol.status == 0:
    #     q = np.hstack((sol.states[0]["q"], sol.states[1]["q"]))
    #     qdot = np.hstack((sol.states[0]["qdot"], sol.states[1]["qdot"]))
    #     u = np.hstack((sol.controls[0]["tau"], sol.controls[1]["tau"]))
    #     t = sol.parameters["time"]
    #     np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_q", q)
    #     np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_qdot", qdot)
    #     np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_u", u)
    #     np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_t", t)
    try:
        # current datetime
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()
        miller.ocp.save(
            sol,
            f"/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/other/"
            f"last_test_min_qdddot",
        )
    except:
        print("not saved")
    print(sol.parameters["time"])
    sol.print()
    sol.graphs(show_bounds=True)
    # sol.animate()
    # sol.animate(nb_frames=-1, show_meshes=True)  # show_mesh=True
    # ma57
    # q = sol.states[0]["q"]
    # qdot = sol.states[0]["qdot"]
    # qddot = sol.controls[0]["qddot"]
    # import biorbd as biorbd
    #
    # m = biorbd.Model(model_path)
    # for qi, qdoti, qddoti in zip(q.T, qdot.T, qddot.T):
    #     print(m.InverseDynamics(qi, qdoti, qddoti).to_array())


if __name__ == "__main__":

    # main("explicit", 8, OdeSolver.RK4(n_integration_steps=5))  # "implicit"  # "explicit"  # "root_explicit"  # "root_implicit")
    # main(MillerDynamics.IMPLICIT, 1,
    #      OdeSolver.RK4(n_integration_steps=5), False)  # "implicit"  # "explicit"  # "root_explicit"  # "root_implicit")
    # main(MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, 1,
    #      OdeSolver.RK4(n_integration_steps=5), True)
    # main(MillerDynamics.EXPLICIT, 1,
    #      OdeSolver.RK4(n_integration_steps=5), False)
    # main(MillerDynamics.ROOT_EXPLICIT, 1,
    #      OdeSolver.RK4(n_integration_steps=5), False)
    # main(MillerDynamics.IMPLICIT, 1,
    #      OdeSolver.RK4(n_integration_steps=5), False)
    # main(MillerDynamics.ROOT_IMPLICIT, 1,
    #      OdeSolver.RK4(n_integration_steps=5), False)
    main(MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, 1, OdeSolver.RK4(n_integration_steps=5), True)
    # main(MillerDynamics.ROOT_IMPLICIT_QDDDOT, 1,
    #      OdeSolver.RK4(n_integration_steps=5), True)
    # "implicit"  # "explicit"  # "root_explicit"  # "root_implicit")
    # main("implicit", 1,
    #      OdeSolver.RK2(n_integration_steps=5))  # "implicit"  # "explicit"  # "root_explicit"  # "root_implicit")
    # main("root_explicit", 8,
    #      OdeSolver.RK4(n_integration_steps=5))  # "implicit"  # "explicit"  # "implicit"  # "root_implicit")