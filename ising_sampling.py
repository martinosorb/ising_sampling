import numpy as np


class IsingModel():
    """A class to sample from arbitrary Ising models.
    The constructor takes the local fields h and connection matrix j.
    The number of units is inferred from their sizes."""
    def __init__(self, h, j):
        h = np.asarray(h)
        j = np.asarray(j)
        # size checks
        numspin = len(h)
        if j.shape != (numspin, numspin):
            raise ValueError('Inconsistent h and J sizes')
        if not np.all(j == j.T):
            raise UserWarning('J is not a symmetric matrix')
        # record the values
        self.j = j
        self.h = h
        self.numspin = numspin

    def hamiltonian(self, state=None):
        """The Ising hamiltonian based on the given h and j."""
        if state is None:
            state = self.spins
        state = np.asarray(state)
        return - np.dot(self.h, state) - 0.5 * np.dot(state,
                                                      np.dot(self.j, state))

    def sample(self, n):
        """Extract n states by Gibbs sampling of the Ising network."""
        # input check
        if n <= 0 or not type(n) == int:
            raise ValueError('n must be a positive integer')
        # results = np.empty([n, self.numspin], dtype=bool)
        # initial state
        self.spins = np.random.choice([True, False], size=self.numspin)
        # results[0] = self.spins
        # iterative loop
        for itern in range(n):
            for spin in range(self.numspin):
                self.spins[spin] = True
                pos_energy = self.hamiltonian()
                self.spins[spin] = False
                neg_energy = self.hamiltonian()
                p = np.exp(neg_energy - pos_energy)
                p /= 1 + p
                if np.random.random() < p:
                    self.spins[spin] = True
            yield self.spins
