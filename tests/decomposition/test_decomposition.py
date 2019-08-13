import os
import unittest

import numpy as np

from pymoo.configuration import get_pymoo
from pymoo.decomposition.perp_dist import PerpendicularDistance
from pymoo.decomposition.weighted_sum import WeightedSum
from tests.test_usage import test_usage


class DecompositionTest(unittest.TestCase):

    def test_decomposition(self):
        folder = os.path.join(get_pymoo(), "pymoo", "usage", "decomposition")
        test_usage([os.path.join(folder, fname) for fname in os.listdir(folder)])

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
        self.assertTrue(np.all(np.abs(np.loadtxt("../resources/perp_dist") - D) < 1e-6))

        D = PerpendicularDistance(_type="cython").do(F, weights, _type="many_to_many")
        self.assertTrue(np.all(np.abs(np.loadtxt("../resources/perp_dist") - D) < 1e-6))



if __name__ == '__main__':
    unittest.main()
