import pyximport
pyximport.install()
from ising_sampling import IsingModel
import numpy as np

numspin = 20
n = 1000

np.random.seed(56426481)
h = np.random.normal(size=numspin)
j = np.random.normal(size=(numspin, numspin))
j += j.T
j[np.diag_indices_from(j)] = np.zeros(numspin)
np.random.seed()

model = IsingModel(h, j)
for x in model.sample(n):
    pass
