import unittest

import numpy as np

from pymoo.tests.test_util import make_individual
from pymoo.util.non_dominated_rank import NonDominatedRank


class NonDominatedRankTest(unittest.TestCase):
    def test_one_dimensional_rank(self):
        i1 = make_individual(np.array([0,0,0]), np.array([1]))
        i2 = make_individual(np.array([1,1,1]), np.array([0]))
        i3 = make_individual(np.array([2,2,2]), np.array([0]))
        i4 = make_individual(np.array([2,2,2]), np.array([0]))
        i5 = make_individual(np.array([3,3,4]), np.array([0]))

        pop = [i1, i2, i3, i4, i5]

        self.assertTrue(np.array_equal(np.array([3, 0, 1, 1, 2]), NonDominatedRank.calc(pop)))
        self.assertTrue(np.array_equal(np.array([3, 0, 1, 1, 2]), NonDominatedRank.calc_from_fronts(NonDominatedRank.calc_as_fronts_pygmo(pop))))


if __name__ == '__main__':
    unittest.main()
