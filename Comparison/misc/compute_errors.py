import numpy as np
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    CostType,
    Shooting,
    Solver,
    SolutionIntegrator,
    InterpolationType,
)


def compute_error_single_shooting(sol, duration: float = 1, integrator: str = None):
    sol_merged = sol.merge_phases()

    if sol_merged.phase_time[-1] < duration:
        raise ValueError(
            f"Single shooting integration duration must be smaller than ocp duration :{sol_merged.phase_time[-1]} s"
        )

    trans_idx = []
    rot_idx = []
    for i in range(sol.ocp.nlp[0].model.nbQ()):
        if sol.ocp.nlp[0].model.nameDof()[i].to_string()[-4:-1] == "Rot":
            rot_idx += [i]
        else:
            trans_idx += [i]
    rot_idx = np.array(rot_idx)
    trans_idx = np.array(trans_idx)

    sol_int = sol.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        merge_phases=True,
        integrator=integrator,
        keep_intermediate_points=False,
    )
    sn_1s = int(sol_int.ns[0] / sol_int.phase_time[-1] * duration)  # shooting node at {duration} second
    if len(rot_idx) > 0:
        single_shoot_error_r = (
            np.sqrt(np.mean((sol_int.states["q"][rot_idx, sn_1s] - sol_merged.states["q"][rot_idx, sn_1s]) ** 2))
            * 180
            / np.pi
        )
    else:
        single_shoot_error_r = np.nan
    if len(trans_idx) > 0:
        single_shoot_error_t = (
            np.sqrt(np.mean((sol_int.states["q"][trans_idx, sn_1s] - sol_merged.states["q"][trans_idx, sn_1s]) ** 2))
            / 1000
        )
    else:
        single_shoot_error_t = np.nan
    return (
        single_shoot_error_t,
        single_shoot_error_r,
    )


def integrate_sol(ocp, sol):
    sol_int = sol.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        integrator=SolutionIntegrator.SCIPY_DOP853,
        keep_intermediate_points=False,
    )
    return sol_int.states["all"]
