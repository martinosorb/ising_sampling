import numpy as np


class IsingModel():
    """A class to sample from arbitrary Ising models.
    The constructor takes the number of spins n, local fields h
    and connection matrix j."""
    def __init__(self, n):
        # size checks
        if type(n) is not int or n <= 0:
            raise ValueError('n must be a positive integer')
        self.numspin = n

    def import_ising01(self, h, j):
        h = np.asarray(h)
        j = np.asarray(j)
        n = self.numspin
        if len(h) != n or j.shape != (n, n):
            raise ValueError('Inconsistent h and j sizes')
        if not np.all(j == j.T):
            raise UserWarning('j is not a symmetric matrix')
        self.hamiltonian = self.__hamiltonian_full
        self.h = h
        self.j = j

    def import_uniform01(self, h, j):
        if not np.size(h) == 1 and np.size(j) == 1:
            raise ValueError('h and j must be scalars')
        self.h = h
        self.j = j
        self.hamiltonian = self.__hamiltonian_mf

    def import_rbm(self, nvis, nhid, visbias, hidbias, vishid):
        visbias = np.asarray(visbias)
        hidbias = np.asarray(hidbias)
        vishid = np.asarray(vishid)
        if type(nvis) is not int or nvis <= 0:
            raise ValueError('nvis must be a positive integer')
        if type(nhid) is not int or nhid <= 0:
            raise ValueError('nhid must be a positive integer')
        if nvis + nhid != self.numspin:
            raise ValueError('nvis+nhid must be equal to the number of units')
        self.nvis, self.nhid = nvis, nhid
        if len(visbias) != nvis or len(hidbias) != nhid:
            raise ValueError('Inconsistent biases size.')
        if vishid.shape != (nvis, nhid):
            raise ValueError('Inconsistent weight matrix shape.\
                             Maybe transpose?')
        self.hamiltonian = self.__hamiltonian_rbm

    # HAMILTONIANS in various forms
    def __hamiltonian_mf(self, state):
        act = np.sum(state)
        return -self.h * act - 0.5 * self.j * act * (act - 1)

    def __hamiltonian_full(self, state):
        return - np.dot(self.h, state) - 0.5 * np.dot(state,
                                                      np.dot(self.j, state))

    def __hamiltonian_rbm(self, state):
        vis = state[:self.nvis]
        hid = state[self.nvis:]
        return - np.dot(vis, self.visbias) - np.dot(hid, self.hidbias) - \
            np.dot(vis, np.dot(hid, self.vishid))

    # Utility functions for computing the Fisher Information tensor
    def __fimfunction_rbm(self, state):
        vis = state[:self.nvis]
        hid = state[self.nvis:]
        prod = np.outer(vis, hid)
        return np.hstack([vis, hid, np.ravel(prod)])

    def sample(self, n):
        """Extract n states by Gibbs sampling of the Ising network."""
        # input check
        if n <= 0 or not type(n) == int:
            raise ValueError('n must be a positive integer')
        # initial state
        self.spins = np.random.choice([True, False], size=self.numspin)
        # iterative loop
        for itern in range(n):
            for spin in range(self.numspin):
                self.spins[spin] = True
                pos_energy = self.hamiltonian(self.spins)
                self.spins[spin] = False
                neg_energy = self.hamiltonian(self.spins)
                p = np.exp(neg_energy - pos_energy)
                p /= 1 + p
                if np.random.random() < p:
                    self.spins[spin] = True
            yield self.spins

    def submodel(self, num):
        # only works with full Ising
        h = self.h[:num]
        j = self.j[:num, :num]
        model = IsingModel(num)
        model.import_ising01(h, j)
        return model
