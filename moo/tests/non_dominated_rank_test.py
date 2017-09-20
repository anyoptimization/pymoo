import unittest

import numpy as np

from moo.tests.test_util import make_individual
from moo.util.non_dominated_rank import NonDominatedRank


class NonDominatedRankTest(unittest.TestCase):
    def test_one_dimensional_rank(self):
        i1 = make_individual(np.array([0]), np.array([1]))
        i2 = make_individual(np.array([1]), np.array([0]))
        i3 = make_individual(np.array([2]), np.array([0]))
        i4 = make_individual(np.array([2]), np.array([0]))
        i5 = make_individual(np.array([3]), np.array([0]))
        self.assertTrue(np.array_equal(np.array([3, 0, 1, 1, 2]), NonDominatedRank.calc([i1, i2, i3, i4, i5])))


if __name__ == '__main__':
    unittest.main()
