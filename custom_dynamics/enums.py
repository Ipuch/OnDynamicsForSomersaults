from enum import Enum, IntEnum


class MillerDynamics(Enum):
    """
    Selection of dynamics to perform the miller ocp
    """

    IMPLICIT = "implicit"
    ROOT_IMPLICIT = "root_implicit"
    EXPLICIT = "explicit"
    ROOT_EXPLICIT = "root_explicit"
    IMPLICIT_TAUDOT_DRIVEN = "implicit_taudot"  # not viable
    IMPLICIT_TAU_DRIVEN_QDDDOT = "implicit_qdddot"  # ok
    ROOT_IMPLICIT_QDDDOT = "root_implicit_qdddot"  # ok
