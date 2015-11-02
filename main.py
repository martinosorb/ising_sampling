import numpy as np
import pyximport
import multiprocessing as mp
pyximport.install()
from ising_sampling import IsingModel

# Sherrington-Kirkpatrick Spin Glass
numspin = 30
n = 5e04
j0 = 10.  # standard deviation

j = np.random.normal(0, 2 * j0, size=(numspin, numspin))
j += j.T
j /= 2.
j[np.diag_indices_from(j)] = np.zeros(numspin)

h = -2 * np.sum(j, axis=1)

model = IsingModel(h, j)
sampled_states = model.sample(n)

states = np.empty([n, numspin])
energies = np.empty(n)

for i, state in enumerate(sampled_states):
    states[i] = state
    energies[i] = model.hamiltonian(state)
