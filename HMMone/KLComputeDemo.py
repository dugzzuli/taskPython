import numpy as np

from HMMone import KLD

mat=np.loadtxt("mat.txt")
print(KLD.symmetricalKL(mat[:, 0], mat[:, 1]))