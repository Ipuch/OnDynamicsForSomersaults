import bioviz
import numpy as np
from bioptim import OptimalControlProgram
import pickle

nb_DoFs = 15

if nb_DoFs == 10:
    model_name = "Model_JeCh_10DoFs.bioMod"
    file = "Model_JeCh_10DoFs.bo"
    # Loading
    ocp, sol = OptimalControlProgram.load(file)
    q = sol.states["q"]
elif nb_DoFs == 15:
    # file_name = "/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/results/miller_explicit_irand/miller_explicit_irand1.pckl"
    file_name = "/home/puchaud/Projets_Python/OnDynamicsForSommersaults_results/other/last_test.bo"
    # model_name = "Model_JeCh_15DoFs.bioMod"
    # file = open(f"{file_name}", "rb")
    # data = pickle.load(file)
    # q = np.hstack((data["states"][0]["q"], data["states"][1]["q"]))
    ocp, sol = OptimalControlProgram.load(file_name)
    sol.graphs()
    sol.animate()
