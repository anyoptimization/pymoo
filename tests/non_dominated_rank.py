import unittest

import numpy as np

from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival


class NonDominatedRankTest(unittest.TestCase):
    def test_crowding_distance(self):
        F = np.array([[0.31, 6.10], [0.22, 7.09], [0.79, 3.97], [0.27, 6.93]])
        cd = RankAndCrowdingSurvival.calc_crowding_distance(F, F_min=np.array([0.1, 0]), F_max=np.array([1, 60]))
        self.assertTrue(np.all(np.round(cd, decimals=2) == np.array([0.63, np.inf, np.inf, 0.12])))

    def test_crowding_distance_degenerated(self):
        F = np.array([[0.0, 6.10], [0.0, 7.09], [0.0, 7.09]])
        cd = RankAndCrowdingSurvival.calc_crowding_distance(F)

        print(cd)

if __name__ == '__main__':
    unittest.main()
