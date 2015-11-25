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
            model = IsingModel([0, 0, 0], [[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            model = IsingModel([0, 0], [[0, 0], [0, 0]])
            list(model.sample(-1))

    def test_res_shapes(self):
        model = IsingModel([0, 0], [[0, 0], [0, 0]])
        res = from_shaped_iter(model.sample(3), bool, (3, 2))
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(len(res[:, 0]), 3)

    def test_shape_null(self):
        model = IsingModel([0], [[0]])
        res = from_shaped_iter(model.sample(1), bool, (1, 1))
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1], 1)

    def test_probabilities_onlyfields(self):
        h = [np.random.normal()]
        j_matrix = [[0]]
        model = IsingModel(h, j_matrix)
        res = from_shaped_iter(model.sample(20000), bool, (20000, 1))
        p1 = 1 / (1 + np.exp(-h[0]))
        obs_prob = np.mean(res, axis=0)[0]
        self.assertAlmostEqual(p1, obs_prob, places=2)

    def test_probabilities_onlyinter(self):
        h = [0, 0]
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        model = IsingModel(h, j_matrix)
        res = from_shaped_iter(model.sample(20000), bool, (20000, 2))
        prod = res[:, 0] * res[:, 1]
        p1 = np.mean(prod)
        self.assertAlmostEqual(p1, 1 / (3 * np.exp(-j) + 1), places=2)

    def test_simple_hamiltonian(self):
        h = np.random.normal(size=2)
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        model = IsingModel(h, j_matrix)
        self.assertAlmostEqual(model.hamiltonian(np.array([0, 0])), 0)
        h = np.array([-1, 1, -1])
        j_matrix = np.array([[0, 1, 2], [1, 0, -1], [2, -1, 0]])
        s = np.array([1, 1, 0])
        a = np.dot(h, s) + .5 * np.dot(s, np.dot(j_matrix, s))
        model = IsingModel(h, j_matrix)
        self.assertAlmostEqual(model.hamiltonian(s), -a)

    def test_submodel(self):
        j = np.random.normal(0, 1, size=(10, 10))
        j += j.T
        j /= 2.
        j[np.diag_indices_from(j)] = np.zeros(10)
        h = np.random.random(size=10)
        model = IsingModel(h, j)
        subm = model.submodel(5)
        h10 = model.hamiltonian(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))
        h5 = subm.hamiltonian(np.array([1, 1, 1, 1, 1]))
        self.assertAlmostEqual(h5, h10)


if __name__ == '__main__':
    unittest.main()
