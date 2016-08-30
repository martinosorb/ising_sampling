import numpy as np
import multiprocessing as mp
from matplotlib.pyplot import gca, cm

class IsingModel():
    """A class to sample from arbitrary Ising models.
    The constructor takes the number of spins n, local fields h
    and connection matrix j."""
    def __init__(self, n):
        # size checks
        if n <= 0:
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
        self.energydiff = self.__energydiff_full
        self.h = h
        self.j = j

    def import_uniform01(self, h, j):
        if not np.size(h) == 1 and np.size(j) == 1:
            raise ValueError('h and j must be scalars')
        self.h = h
        self.j = j
        self.hamiltonian = self.__hamiltonian_mf
        self.energydiff = self.__energydiff_mf

    def import_uniformPM(self, h, j):
        if not np.size(h) == 1 and np.size(j) == 1:
            raise ValueError('h and j must be scalars')
        self.h = 2*(h - (self.numspin-1)*j)  # sure about -1?
        self.j = 2*j
        self.hamiltonian = self.__hamiltonian_mf
        self.energydiff = self.__energydiff_mf

    def import_2d01(self, h, j, shape):
        if np.size(h) != 1 or np.size(j) != 1:
            raise ValueError("h and j must be scalars")
        if shape[0]*shape[1] != self.numspin:
            raise ValueError("The shape does not match the number of spins")
        self.shape = shape
        self.h, self.j = h, j
        self.hamiltonian = self.__hamiltonian_notimplemented
        self.energydiff = self.__energydiff_2d

# TODO it should also check the diagonal is empty
    def import_isingPM(self, h, j):
        h = np.asarray(h)
        j = np.asarray(j)
        n = self.numspin
        if len(h) != n or j.shape != (n, n):
            raise ValueError('Inconsistent h and j sizes')
        if not np.all(j == j.T):
            raise UserWarning('j is not a symmetric matrix')
        h01 = 2*(h - np.sum(j, axis=1))
        j01 = 2*j
        self.hamiltonian = self.__hamiltonian_full
        self.energydiff = self.__energydiff_full
        self.h = h01
        self.j = j01

    def import_rbm01(self, nvis, nhid, visbias, hidbias, vishid):
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
        if visbias.size != nvis or hidbias.size != nhid:
            raise ValueError('Inconsistent biases size.')
        if vishid.shape != (nvis, nhid):
            raise ValueError('Inconsistent weight matrix shape.\
                             Maybe transpose?')
        # self.visbias, self.hidbias, self.vishid = visbias, hidbias, vishid.T
        # self.hamiltonian = self.__hamiltonian_rbm
        # self.energydiff = self.__energydiff_rbm
        self.h = np.hstack([visbias, hidbias])
        self.j = np.zeros([self.numspin, self.numspin])
        self.j[nvis:, :nvis] = vishid.T
        self.j += self.j.T
        self.hamiltonian = self.__hamiltonian_full
        self.energydiff = self.__energydiff_full

    # HAMILTONIANS in various forms
    def __energydiff_mf(self, state, i):
        state[i] = False
        return self.h + self.j * np.sum(state)

    def __hamiltonian_mf(self, state):
        act = np.sum(state)
        return -self.h * act - 0.5 * self.j * act * (act - 1)

    def __energydiff_full(self, state, i):
        return self.h[i] + np.dot(self.j[i], state)

    def __hamiltonian_full(self, state):
        return - np.dot(self.h, state) - 0.5 * np.dot(state,
                                                      np.dot(self.j, state))

    def __energydiff_2d(self, state, i):
        a, b = self.shape
        x = i % a
        y = i // a
        # to invert: i = y*a + x
        u = ((y + 1) % b) * a + x
        d = ((y - 1) % b) * a + x
        r = y*a + (x + 1) % a
        l = y*a + (x - 1) % a
        return self.h + self.j * (state[u] + state[d] + state[l] + state[r])

    def __hamiltonian_notimplemented(self, state):
        raise NotImplementedError()

    # def __energydiff_rbm(self, state, i):
    #     pass

    # def __hamiltonian_rbm(self, state):
    #     vis = state[:self.nvis]
    #     hid = state[self.nvis:]
    #     return - np.dot(vis, self.visbias) - np.dot(hid, self.hidbias) - \
    #         np.dot(vis, np.dot(hid, self.vishid))

    # Utility functions for computing the Fisher Information tensor
    def fimfunction_rbm(self, state):
        vis = state[:self.nvis]
        hid = state[self.nvis:]
        prod = np.outer(vis, hid)
        return np.hstack([vis, hid, np.ravel(prod)])

    def fimfunction_pairwise(self, state):
        prod_pairs = np.outer(state, state)[np.triu_indices(self.numspin, 1)]
        return np.hstack([state, np.ravel(prod_pairs)])

    def fisher_information(self, sample, fimfunction):
        # it should accept either a sample given as an array
        # for example one saved before, or a generator given
        # by the 'sample' method
        s = [fimfunction(x) for x in sample]
        return np.cov(s, rowvar=0)

    def random_state(self):
        self.spins = np.random.choice([True, False], size=self.numspin)

    def sample(self, n, beta=1):
        """Extract n states by Gibbs sampling of the Ising network."""
        # input check
        if n <= 0 or not type(n) == int:
            raise ValueError('n must be a positive integer')
        # initial state
        self.random_state()
        # iterative loop
        for itern in range(n):
            for spin in range(self.numspin):
                delta_e = self.energydiff(self.spins, spin) * beta
                p = np.exp(delta_e)
                self.spins[spin] = False
                p /= 1 + p
                if np.random.random() < p:
                    self.spins[spin] = True
            yield self.spins

    def __sample_function(argtuple):
        (samplefunc, beta, N, function) = argtuple
        sample = samplefunc(N, beta=beta)
        return [function(x) for x in sample]

    def sample_function_at_betas(self, betas, N, function, parallel=1):
        raise ImportError('Not implemented')
        if parallel > 1:
            p = mp.Pool(parallel)
            args = [(self.sample, beta, N, function) for beta in betas]
            results = p.map(self.__sample_function, args)
            p.close()
        else:
            results = list(map(sample_function, betas))
        return results

    def submodel(self, num):
        if num > self.numspin:
            raise ValueError('Submodel size must be smaller than model size.')
        # don't use with RBM
        model = IsingModel(num)
        if np.size(self.h) == 1:
            model.import_uniform01(self.h, self.j)
        else:
            h = self.h[:num]
            j = self.j[:num, :num]
            model.import_ising01(h, j)
        return model

    def show(self, ax=None):
        if ax is None:
            ax = gca()
        ax.pcolormesh(self.spins.reshape(self.shape), cmap=cm.gray)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
