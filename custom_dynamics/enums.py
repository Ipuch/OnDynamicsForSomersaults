from enum import Enum


class MillerDynamics(Enum):
    """
    Selection of dynamics to perform the miller ocp
    """

    EXPLICIT = "explicit"
    ROOT_EXPLICIT = "root_explicit"
    IMPLICIT = "implicit"
    ROOT_IMPLICIT = "root_implicit"
    IMPLICIT_TAU_DRIVEN_QDDDOT = "implicit_qdddot"
    ROOT_IMPLICIT_QDDDOT = "root_implicit_qdddot"
