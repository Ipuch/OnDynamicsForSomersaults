from enum import Enum, IntEnum


class Model(Enum):
    """
    Models path
    """

    my_path = "Comparison/models/"
    ACROBOT = my_path + "acrobot.bioMod"
    CARTPOLE = my_path + "cart_pole.bioMod"
    FUTURA_PENDULUM = my_path + "futura_pendulum.bioMod"
    INERTIA_WHEEL_PENDULUM = my_path + "inertia_wheel_pendulum.bioMod"
    TRIPLE_PENDULUM = my_path + "triple_pendulum.bioMod"
