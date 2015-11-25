from ising_sampling import IsingModel
import numpy as np
from time import time

numspin = 30
n = 4000
seed = np.random.choice(1000)

j0 = np.random.random()
h0 = np.random.normal()

j = np.ones((numspin, numspin)) * j0
h = np.ones(numspin) * h0
j[np.diag_indices_from(j)] = np.zeros(numspin)

t = time()
model_full = IsingModel(numspin, h, j)
np.random.seed(seed)
for x in model_full.sample(n):
    pass

print(time() - t)
t = time()

model_mf = IsingModel(numspin, h0, j0)
np.random.seed(seed)
for x in model_mf.sample(n):
    pass

print(time() - t)
