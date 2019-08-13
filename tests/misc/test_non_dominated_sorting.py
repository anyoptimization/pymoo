import unittest

import numpy as np

from pymoo.util.function_loader import load_function


class FastNonDominatedSortTest(unittest.TestCase):

    def test_non_dominated_sorting(self):
        F = np.random.random((100,2))
        fronts = load_function("fast_non_dominated_sort", _type="python")(F)
        fronts = [np.sort(fronts[k]) for k in range(len(fronts))]

        _fronts = load_function("fast_non_dominated_sort", _type="cython")(F)
        _fronts = [np.sort(_fronts[k]) for k in range(len(_fronts))]

        self.assertEqual(len(fronts), len(_fronts))

        for k in range(len(_fronts)):
            is_equal = _fronts[k] == fronts[k]
            self.assertTrue(np.all(is_equal))

if __name__ == '__main__':
    unittest.main()
