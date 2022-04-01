import bioviz
import numpy as np
from bioptim import OptimalControlProgram
import pickle
import matplotlib.pyplot as plt

nb_DoFs = 15

if nb_DoFs == 10:
    model_name = "Model_JeCh_10DoFs.bioMod"
    file = "Model_JeCh_10DoFs.bo"
    # Loading
    ocp, sol = OptimalControlProgram.load(file)
    q = sol.states["q"]
elif nb_DoFs == 15:
    # file_name = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/miller_explicit_irand/miller_explicit_irand1.pckl"
    file_name1 = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/other/last_test_min_qdddot.bo"
    # model_name = "Model_JeCh_15DoFs.bioMod"
    # file = open(f"{file_name}", "rb")
    # data = pickle.load(file)
    # q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))
    ocp, sol = OptimalControlProgram.load(file_name1)
    data = sol.states
    q = np.hstack((data[0]["q"], data[1]["q"]))

    plt.plot(q[10, :], label="qdddot")

    file_name2 = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/raw_with_min_qddot/miller_root_explicit_irand49_extraobjFalse_125_25.pckl"
    model_name = "Model_JeCh_15DoFs.bioMod"
    file = open(f"{file_name2}", "rb")
    data = pickle.load(file)
    q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))
    # ocp, sol = OptimalControlProgram.load(file_name2)
    # data = sol.states
    # q = np.hstack((data[0]["q"], data[1]["q"]))

    plt.plot(q[10, :], label="exp")
    plt.legend()
    plt.show()
