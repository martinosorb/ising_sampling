import numpy as np
import multiprocessing as mp
from ising_sampling import IsingModel

numspin = 80
n = 10000

resdir = 'results_mf_j_1overN_h_minushalf/'

# Sherrington-Kirkpatrick Spin Glass
# j0 = 1.  # standard deviation
# j = np.random.normal(0, 2 * j0, size=(numspin, numspin))
# j += np.copy(j.T)
# j /= 2.
# j[np.diag_indices_from(j)] = np.zeros(numspin)
# h = -2 * np.sum(j, axis=1)

# Mean Field
j = 1 / n
h = -.5

np.save(resdir + "h.npy", h)
np.save(resdir + "j.npy", j)


def sample_e_with_beta(beta):
    model = IsingModel(n, beta * h, beta * j)
    sampled_states = model.sample(n)

    states = np.empty([n, numspin], dtype=bool)
    energies = np.empty(n)

    for i, state in enumerate(sampled_states):
        states[i] = state
        energies[i] = model.hamiltonian(state.astype(int))

    np.save(resdir + "states_" + "beta" + str(beta) + "_n" +
            str(numspin) + ".npy", states)
    np.save(resdir + "energies_" + "beta" + str(beta) + "_n" +
            str(numspin) + ".npy", energies)


P = mp.Pool()
P.map(sample_e_with_beta, np.linspace(2., 6., 11))
