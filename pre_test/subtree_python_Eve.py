
import biorbd
import numpy as np

m = biorbd.Model("/home/user/Documents/Programmation/Eve/Tests_NoteTech_Pierre/hyq.biomod")
subtree = [ [] for i in range(m.nbQ())]
for i in range(m.nbQ()):
    subtree[i].append(i)
    for j in range(m.nbQ()):
        if m.lambda_q(j) in subtree[i]:
            subtree[i].append(j)













