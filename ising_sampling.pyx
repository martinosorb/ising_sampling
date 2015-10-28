import numpy as np
import cython
cimport cython
cimport numpy as np


cdef class IsingModel:
    """A class to sample from arbitrary Ising models.
    The constructor takes the local fields h and connection matrix j.
    The number of units is inferred from their sizes."""

    cpdef np.float64_t[:] h
    cpdef np.float64_t[:, :] j
    cdef int numspin

    def __init__(self, nh, nj):
        global h, j
        nh = np.asarray(nh)
        nj = np.asarray(nj)
        # size checks
        numspin = len(nh)
        if nj.shape != (numspin, numspin):
            raise ValueError('Inconsistent h and J sizes')
        if not np.all(nj == nj.T):
            raise UserWarning('J is not a symmetric matrix')
        # record the values
        self.j = nj.copy().astype(np.float64)
        self.h = nh.copy().astype(np.float64)
        self.numspin = numspin

    # def hamiltonian(self, state=None):
    #     """The Ising hamiltonian based on the given h and j."""
    #     if state is None:
    #         state = self.spins
    #     state = np.asarray(state)
    #     return - self.h @ state - 0.5 * state @ self.j @ state

    @cython.boundscheck(False)
    cpdef np.float64_t hamiltonian(self, np.ndarray[np.int64_t, ndim=1] state):
        """The Ising hamiltonian based on the given h and j."""
        h = self.h
        j = self.j
        cdef np.float64_t out = 0.
        cdef np.int64_t i, k
        cdef np.float64_t eff_field = 0.
        #out = - h @ state - 0.5 * state @ j @ state
        for i in range(len(h)):
            eff_field = 0.
            for k in range(i):
                eff_field += j[i, k] * state[k]
            out -= state[i] * (h[i] + eff_field)
        return out

    def sample(self, n):
        """Extract n states by Gibbs sampling of the Ising network."""
        # input check
        if n <= 0 or not type(n) == int:
            raise ValueError('n must be a positive integer')
        results = np.empty([n, self.numspin], dtype=bool)
        # initial state
        spins = np.random.choice([1, 0], size=self.numspin)
        results[0] = spins.astype(bool)
        # iterative loop
        for itern in range(n):
            for spin in range(self.numspin):
                spins[spin] = 1
                pos_energy = self.hamiltonian(spins)
                spins[spin] = 0
                neg_energy = self.hamiltonian(spins)
                p = np.exp(neg_energy - pos_energy)
                p /= 1 + p
                if np.random.random() < p:
                    spins[spin] = 1
            results[itern] = spins.astype(bool)

        return results
