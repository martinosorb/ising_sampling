import numpy as np
import pyximport
import multiprocessing as mp
pyximport.install()
from ising_sampling import IsingModel

# Sherrington-Kirkpatrick Spin Glass
numspin = 80
n = 10000
j0 = 1.  # standard deviation

j = np.random.normal(0, 2 * j0, size=(numspin, numspin))
j += np.copy(j.T)
j /= 2.
j[np.diag_indices_from(j)] = np.zeros(numspin)

h = -2 * np.sum(j, axis=1)

np.save("results/h.npy", h)
np.save("results/j.npy", j)


def sample_e_with_beta(beta):
    model = IsingModel(beta * h, beta * j)
    sampled_states = model.sample(n)

    states = np.empty([n, numspin], dtype=bool)
    energies = np.empty(n)

    for i, state in enumerate(sampled_states):
        states[i] = state
        energies[i] = model.hamiltonian(state.astype(int))

    np.save("results/states_" + "beta" + str(beta) + "_n" +
            str(numspin) + ".npy", states)
    np.save("results/energies_" + "beta" + str(beta) + "_n" +
            str(numspin) + ".npy", energies)


P = mp.Pool()
P.map(sample_e_with_beta, np.linspace(0.0, 2., 10))
