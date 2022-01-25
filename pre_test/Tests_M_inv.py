

import biorbd
import numpy as np

m = biorbd.Model("/home/user/Documents/Programmation/Eve/OnDynamicsForSommersaults/Model_JeCh_10DoFs.bioMod")

for i in range(10):
    Q = np.reshape(np.random.random((1, 10)), (10, ))
    mass_matrix = m.massMatrix(Q).to_array()
    print(mass_matrix)
    print('\n')






