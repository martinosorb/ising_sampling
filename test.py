import unittest
from ising_sampling import IsingModel
import numpy as np

# I want a function sample_ising(h, J, N)
# which gives me N samples of an Ising model
# at local fields h and interactions J


class TestIsingModel(unittest.TestCase):

    def test_input_data(self):
        with self.assertRaises(ValueError):
            model = IsingModel([0, 0, 0], [[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            model = IsingModel([0, 0], [[0, 0], [0, 0]])
            model.sample(-1)

    def test_res_shapes(self):
        model = IsingModel([0, 0], [[0, 0], [0, 0]])
        res = model.sample(3)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(len(res[:, 0]), 3)

    def test_shape_null(self):
        model = IsingModel([0], [[0]])
        res = model.sample(1)
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1], 1)

    def test_probabilities_onlyfields(self):
        h = [np.random.normal()]
        j_matrix = [[0]]
        model = IsingModel(h, j_matrix)
        res = model.sample(20000)
        p1 = 1 / (1 + np.exp(-h[0]))
        obs_prob = np.mean(res, axis=0)[0]
        self.assertAlmostEqual(p1, obs_prob, places=2)

    def test_probabilities_onlyinter(self):
        h = [0, 0]
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        model = IsingModel(h, j_matrix)
        res = model.sample(20000)
        prod = res[:, 0] * res[:, 1]
        p1 = np.mean(prod)
        self.assertAlmostEqual(p1, 1 / (3 * np.exp(-j) + 1), places=2)

    def test_simple_hamiltonian(self):
        h = np.random.normal(size=2)
        j = np.random.normal()
        j_matrix = [[0, j], [j, 0]]
        model = IsingModel(h, j_matrix)
        self.assertAlmostEqual(model.hamiltonian([False, False]), 0)
        model.h = np.array([2, 2, 2])
        model.j = np.array([[0, 1, -2], [1, 0, -1], [-2, -1, 0]])
        self.assertAlmostEqual(model.hamiltonian([True, False, True]), -2.)


if __name__ == '__main__':
    unittest.main()
