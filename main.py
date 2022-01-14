import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver
from miller_ocp import MillerOcp


def main():
    n_shooting = 150
    ode_solver = OdeSolver.RK4(n_integration_steps=5)
    duration = 1.545
    n_threads = 8
    model_path = "Model_JeCh_10DoFs.bioMod"
    # --- Solve the program --- #
    miller = MillerOcp(
        biorbd_model_path=model_path,
        duration=duration,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        implicit_dynamics=False,
        n_threads=n_threads,
        vertical_velocity_0=9.2,
        somersaults=4 * np.pi,
        twists=4 * np.pi,
    )

    miller.ocp.add_plot_penalty(CostType.ALL)

    solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solv.set_print_level(5)
    sol = miller.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    # sol.print()
    sol.animate()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()

