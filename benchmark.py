from ising_sampling import IsingModel
import numpy as np

numspin = 30
n = 4000

np.random.seed(56426481)
h = np.random.normal(size=numspin)
j = np.random.normal(size=(numspin, numspin))
j += j.T
j[np.diag_indices_from(j)] = np.zeros(numspin)
np.random.seed()

model = IsingModel(numspin)
model.import_ising01(h, j)

for x in model.sample(n):
    pass
