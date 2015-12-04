import unittest
from ising_sampling import IsingModel
import numpy as np

# I want a function sample_ising(h, J, N)
# which gives me N samples of an Ising model
# at local fields h and interactions J


def from_shaped_iter(iterable, dtype, shape):
    a = np.empty(shape, dtype)
    for i, x in enumerate(iterable):
        a[i] = x
    return a


class TestIsingModel(unittest.TestCase):

    def test_input_data(self):
        with self.assertRaises(ValueError):
            model = IsingModel(3)
            model.import_ising01([0, 0, 0], [[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            model = IsingModel(2)
            model.import_ising01([0, 0, 0], [[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            model = IsingModel(0)
        with self.assertRaises(ValueError):
            model = IsingModel(2)
            list(model.sample(-1))

    def test_res_shapes(self):
        model = IsingModel(2)
        model.import_ising01([0, 0], [[0, 0], [0, 0]])
        res = from_shaped_iter(model.sample(3), bool, (3, 2))
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(len(res[:, 0]), 3)

    def test_shape_null(self):
        model = IsingModel(1)
        model.import_ising01([0], [[0]])
        res = from_shaped_iter(model.sample(1), bool, (1, 1))
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1], 1)

    def test_probabilities_onlyfields(self):
        h = [np.random.normal()]
        j_matrix = [[0]]
        model = IsingModel(1)
        model.import_ising01(h, j_matrix)
        res = from_shaped_iter(model.sample(20000), bool, (20000, 1))
        p1 = 1 / (1 + np.exp(-h[0]))
        obs_prob = np.mean(res, axis=0)[0]
        self.assertAlmostEqual(p1, obs_prob, places=2)

    def test_probabilities_onlyinter(self):
        h = [0, 0]
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        model = IsingModel(2)
        model.import_ising01(h, j_matrix)
        res = from_shaped_iter(model.sample(20000), bool, (20000, 2))
        prod = res[:, 0] * res[:, 1]
        p1 = np.mean(prod)
        self.assertAlmostEqual(p1, 1 / (3 * np.exp(-j) + 1), places=2)

    def test_simple_hamiltonian(self):
        h = np.random.normal(size=2)
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        model = IsingModel(2)
        model.import_ising01(h, j_matrix)
        self.assertAlmostEqual(model.hamiltonian(np.array([0, 0])), 0)
        h = np.array([-1, 1, -1])
        j_matrix = np.array([[0, 1, 2], [1, 0, -1], [2, -1, 0]])
        s = np.array([1, 1, 0])
        a = np.dot(h, s) + .5 * np.dot(s, np.dot(j_matrix, s))
        model = IsingModel(3)
        model.import_ising01(h, j_matrix)
        self.assertAlmostEqual(model.hamiltonian(s), -a)

    def test_mf_hamiltonian(self):
        h = np.random.normal()
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        h_vector = [h, h]
        full_model = IsingModel(2)
        mf_model = IsingModel(2)
        full_model.import_ising01(h_vector, j_matrix)
        mf_model.import_uniform01(h, j)
        state = np.random.choice([True, False], size=2)
        self.assertAlmostEqual(full_model.hamiltonian(state),
                               mf_model.hamiltonian(state))

    def test_submodel(self):
        j = np.random.normal(0, 1, size=(10, 10))
        j += j.T
        j /= 2.
        j[np.diag_indices_from(j)] = np.zeros(10)
        h = np.random.random(size=10)
        model = IsingModel(10)
        model.import_ising01(h, j)
        subm = model.submodel(5)
        inp = [True, True, True, True, True, False, False, False, False, False]
        h10 = model.hamiltonian(inp)
        h5 = subm.hamiltonian(inp[:5])
        self.assertAlmostEqual(h5, h10)

    def test_rbm_fim(self):
        j = np.random.normal(0, 1)
        visbias = np.random.normal()
        hidbias = np.random.normal()

        model = IsingModel(2)
        model.import_rbm(1, 1, [visbias], [hidbias], [[j]])

        z = (np.exp(visbias) + np.exp(hidbias) +
             np.exp(visbias + hidbias + j) + 1)
        f = lambda x: x / z * (1 - x / z)
        exp0 = np.exp(visbias + hidbias + j)
        exp1 = np.exp(visbias) + exp0
        exp2 = np.exp(hidbias) + exp0

        varvis = f(exp1)
        varhid = f(exp2)
        varprod = f(exp0)
        cov_vishid = exp0 / z - exp1 * exp2 / z ** 2
        cov_visprod = exp0 / z * (1 - exp1 / z)
        cov_hidprod = exp0 / z * (1 - exp2 / z)

        sample = model.sample(20000)
        fim = model.fisher_information(sample, model.fimfunction_rbm)

        self.assertAlmostEqual(fim[0, 0], varvis, places=2)
        self.assertAlmostEqual(fim[1, 1], varhid, places=2)
        self.assertAlmostEqual(fim[2, 2], varprod, places=2)
        self.assertAlmostEqual(fim[1, 0], cov_vishid, places=2)
        self.assertAlmostEqual(fim[2, 0], cov_visprod, places=2)
        self.assertAlmostEqual(fim[1, 2], cov_hidprod, places=2)

    def test_diff(self):
        h = np.random.normal(size=2)
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        model = IsingModel(2)
        model.import_ising01(h, j_matrix)

        spins = np.random.choice([True, False], size=2)
        spin = np.random.choice([0, 1])
        spins[spin] = True
        pos_energy = model.hamiltonian(spins)
        spins[spin] = False
        neg_energy = model.hamiltonian(spins)
        p = neg_energy - pos_energy

        self.assertAlmostEqual(p, model.energydiff_full(spins, spin))


if __name__ == '__main__':
    unittest.main()
