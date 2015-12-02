import numpy as np


class IsingModel():
    """A class to sample from arbitrary Ising models.
    The constructor takes the number of spins n, local fields h
    and connection matrix j."""
    def __init__(self, n, h, j):
        self.h = np.asarray(h)
        self.j = np.asarray(j)
        # size checks
        if type(n) is not int or n <= 0:
            raise ValueError('n must be a positive integer')
        self.numspin = n
        if np.size(self.h) == 1:
            if np.size(self.j) != 1:
                raise ValueError('Inconsistent h and j sizes')
            self.hamiltonian = self.__hamiltonian_mf
        else:
            if len(self.h) != n or self.j.shape != (n, n):
                raise ValueError('Inconsistent h and j sizes')
            self.hamiltonian = self.__hamiltonian_full
        if not np.all(self.j == self.j.T):
            raise UserWarning('j is not a symmetric matrix')

    # @cachefunc
    def __hamiltonian_mf(self, state):
        act = np.sum(state)
        return -self.h * act - 0.5 * self.j * act * (act - 1)

    def __hamiltonian_full(self, state):
        return - np.dot(self.h, state) - 0.5 * np.dot(state,
                                                      np.dot(self.j, state))

    def sample(self, n):
        """Extract n states by Gibbs sampling of the Ising network."""
        # if np.size(self.h) == 1:
        #     hamiltonian = self.__hamiltonian_mf
        # else:
        #     hamiltonian = self.__hamiltonian_full
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
        if np.size(self.h) == 1:
            return self
        h = self.h[:num]
        j = self.j[:num, :num]
        return IsingModel(num, h, j)


class Rbm(IsingModel):
    def __init__(self, nvis, nhid, visbias, hidbias, vishid):
        visbias = np.asarray(visbias)
        hidbias = np.asarray(hidbias)
        vishid = np.asarray(vishid)
        if type(nvis) is not int or nvis <= 0:
            raise ValueError('nvis must be a positive integer')
        if type(nhid) is not int or nhid <= 0:
            raise ValueError('nhid must be a positive integer')
        self.numspin = nvis + nhid
        self.nvis, self.nhid = nvis, nhid
        if len(visbias) != nvis or len(hidbias) != nhid:
            raise ValueError('Inconsistent biases size.')
        if vishid.shape != (nvis, nhid):
            raise ValueError('Inconsistent weight matrix shape.\
                             Maybe transpose?')
        self.hamiltonian = self.__hamiltonian_rbm

    def submodel(self, num):
        raise NotImplementedError()

    def __hamiltonian_rbm(self, state):
        vis = state[:self.nvis]
        hid = state[self.nvis:]
        return - np.dot(vis, self.visbias) - np.dot(hid, self.hidbias) - \
            np.dot(vis, np.dot(hid, self.vishid))
