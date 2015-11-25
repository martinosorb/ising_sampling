import numpy as np


class IsingModel():
    """A class to sample from arbitrary Ising models.
    The constructor takes the number of spins n, local fields h
    and connection matrix j."""
    def __init__(self, n, h, j):
        h = np.asarray(h)
        j = np.asarray(j)
        # size checks
        if type(n) is not int or n <= 0:
            raise ValueError('n must be a positive integer')
        self.numspin = n
        if np.size(h) == 1:
            if np.size(j) != 1:
                raise ValueError('Inconsistent h and j sizes')
        else:
            if len(h) != n or j.shape != (n, n):
                raise ValueError('Inconsistent h and j sizes')
        if not np.all(j == j.T):
            raise UserWarning('J is not a symmetric matrix')
        # record the values
        self.j = j
        self.h = h

    def hamiltonian(self, state):
        """The Ising hamiltonian based on the given h and j."""
        state = np.asarray(state)
        if np.size(self.h) == 1:
            return self.__hamiltonian_mf(state)
        return self.__hamiltonian_full(state)

    # @cachefunc
    def __hamiltonian_mf(self, state):
        act = np.sum(state)
        return -self.h * act - 0.5 * self.j * act * (act - 1)

    def __hamiltonian_full(self, state):
        return - np.dot(self.h, state) - 0.5 * np.dot(state,
                                                      np.dot(self.j, state))

    def sample(self, n):
        """Extract n states by Gibbs sampling of the Ising network."""
        if np.size(self.h) == 1:
            hamiltonian = self.__hamiltonian_mf
        else:
            hamiltonian = self.__hamiltonian_full
        # input check
        if n <= 0 or not type(n) == int:
            raise ValueError('n must be a positive integer')
        # initial state
        self.spins = np.random.choice([True, False], size=self.numspin)
        # iterative loop
        for itern in range(n):
            for spin in range(self.numspin):
                self.spins[spin] = True
                pos_energy = hamiltonian(self.spins)
                self.spins[spin] = False
                neg_energy = hamiltonian(self.spins)
                p = np.exp(neg_energy - pos_energy)
                p /= 1 + p
                if np.random.random() < p:
                    self.spins[spin] = True
            yield self.spins

    def submodel(self, num):
        h = self.h[:num]
        j = self.j[:num, :num]
        return IsingModel(num, h, j)
