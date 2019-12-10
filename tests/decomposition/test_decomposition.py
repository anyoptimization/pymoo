import unittest

import numpy as np

from pymoo.decomposition.perp_dist import PerpendicularDistance
from pymoo.decomposition.weighted_sum import WeightedSum
from tests import path_to_test_resources


class DecompositionTest(unittest.TestCase):

    def test_one_to_one(self):
        F = np.random.random((2, 2))
        weights = np.array([[0.5, 0.5], [0.25, 0.25]])
        val = WeightedSum().do(F, weights=weights)

        self.assertEqual(val.ndim, 1)
        self.assertEqual(val.shape[0], 2)

    def test_one_to_many(self):
        F = np.random.random((1, 2))
        weights = np.array([[0.5, 0.5], [0.25, 0.25]])
        val = WeightedSum().do(F, weights=weights)

        self.assertEqual(val.ndim, 1)
        self.assertEqual(val.shape[0], 2)

    def test_many_to_one(self):
        F = np.random.random((10, 2))
        weights = np.array([[0.5, 0.5]])
        val = WeightedSum().do(F, weights=weights)

        self.assertEqual(val.ndim, 1)
        self.assertEqual(val.shape[0], 10)

    def test_many_to_many(self):
        F = np.random.random((10, 2))
        weights = np.array([[0.5, 0.5], [0.25, 0.25]])
        val = WeightedSum().do(F, weights=weights)

        self.assertEqual(val.shape[0], 10)
        self.assertEqual(val.shape[1], 2)

    def test_perp_dist(self):
        np.random.seed(1)
        F = np.random.random((100, 3))
        weights = np.random.random((10, 3))

        D = PerpendicularDistance(_type="python").do(F, weights, _type="many_to_many")
        np.testing.assert_allclose(D, np.loadtxt(path_to_test_resources("perp_dist")))

        D = PerpendicularDistance(_type="cython").do(F, weights, _type="many_to_many")
        np.testing.assert_allclose(D, np.loadtxt(path_to_test_resources("perp_dist")))


if __name__ == '__main__':
    unittest.main()
