import numpy as np
from bioptim import OdeSolver, CostType
from bioptim import Solver
from miller_ocp import MillerOcp


def main():
    n_shooting = (125, 25)
    ode_solver = OdeSolver.RK4(n_integration_steps=5)
    duration = 1.545
    n_threads = 3
    model_path = "Model_JeCh_15DoFs.bioMod"
    dynamics_type = "implicit"  # "implicit"  # "explicit"  # "root_explicit"  # "root_implicit"
    # mettre une contrainte
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
    )

    miller.ocp.print(to_console=True)

    miller.ocp.add_plot_penalty(CostType.ALL)
    np.random.seed(0)

    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(1000)
    solver.set_print_level(5)

    sol = miller.ocp.solve(solver)

    # --- Show results --- #
    if sol.status == 0:
        q = np.hstack((sol.states[0]['q'], sol.states[1]['q']))
        qdot = np.hstack((sol.states[0]['qdot'], sol.states[1]['qdot']))
        u = np.hstack((sol.controls[0]['tau'], sol.controls[1]['tau']))
        t = sol.parameters['time']
        np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_q", q)
        np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_qdot", qdot)
        np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_u", u)
        np.save(f"/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/raw/27jan_4_t", t)

    sol.print()
    # sol.graphs()
    # sol.animate()
    sol.animate(nb_frames=-1, show_meshes=True)  # show_mesh=True
    # ma57
    q = sol.states[0]["q"]
    qdot = sol.states[0]["qdot"]
    qddot = sol.controls[0]["qddot"]
    import biorbd as biorbd

    m = biorbd.Model(model_path)
    for qi, qdoti, qddoti in zip(q.T, qdot.T, qddot.T):
        print(m.InverseDynamics(qi, qdoti, qddoti).to_array())


if __name__ == "__main__":
    main()
