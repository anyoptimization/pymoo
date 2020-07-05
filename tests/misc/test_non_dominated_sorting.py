import unittest

import numpy as np

from pymoo.util.function_loader import load_function


class FastNonDominatedSortTest(unittest.TestCase):
    def assertFrontEqual(self, fronts_a, fronts_b):
        self.assertEqual(len(fronts_a), len(fronts_b))
        for a, b in zip(fronts_a, fronts_b):
            self.assertEqual(len(a), len(b))
            self.assertEqual(set(a), set(b))

    def test_fast_non_dominated_sorting(self):
        F = np.random.random((100,2))
        fronts = load_function("fast_non_dominated_sort", _type="python")(F)
        _fronts = load_function("fast_non_dominated_sort", _type="cython")(F)

        self.assertFrontEqual(fronts, _fronts)

    def test_efficient_non_dominated_sort(self):
        print("Testing ENS...")
        F = np.ones((1000, 3))
        F[:, 1:] = np.random.random((1000, 2))
        _fronts = load_function("fast_non_dominated_sort", _type="python")(F)
        fronts = load_function("efficient_non_dominated_sort", _type="python")(F)

        self.assertFrontEqual(_fronts, fronts)

        fronts = load_function("efficient_non_dominated_sort", _type="python")(F, strategy="binary")
        self.assertFrontEqual(_fronts, fronts)

    def test_tree_based_non_dominated_sort(self):
        print("Testing T-ENS...")
        F = np.ones((1000, 3))
        F[:, 1:] = np.random.random((1000, 2))
        _fronts = load_function("fast_non_dominated_sort", _type="python")(F)

        fronts = load_function("tree_based_non_dominated_sort", _type="python")(F)
        self.assertFrontEqual(_fronts, fronts)


if __name__ == '__main__':
    unittest.main()
