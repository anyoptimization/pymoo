import os
import pickle
import unittest

import numpy as np

from pymoo.model.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival
from pymoo.rand.impl.custom_random_generator import CustomRandomGenerator
from pymoo.util.non_dominated_rank import NonDominatedRank


class NSGA2Test(unittest.TestCase):
    """
    Some methods compare with one test run from the C code (200 generations) and
    tests if the rank is always equal.

    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join("resources", "cnsga2_zdt4.dat"), 'rb') as f:
            cls.data = pickle.load(f)

    # tests whether the first number by the random generator is equal
    def test_custom_random_generator(self):
        rand = CustomRandomGenerator(0.1)
        val = rand.rand()
        self.assertAlmostEqual(val, 0.337237, places=6)

    def test_non_dominated_rank(self):
        for i, D in enumerate(self.data):
            _rank = NonDominatedRank().calc(D['F'])
            rank = D['rank'].astype(np.int) - 1
            is_equal = np.all(_rank == rank)
            self.assertTrue(is_equal)

    def test_crowding_distance(self):
        for i, D in enumerate(self.data):

            _rank = np.full(D['crowding'].shape[0], -1)
            _crowding = np.full(D['crowding'].shape[0], -1.0)
            fronts = NonDominatedRank.calc_as_fronts(D['F'], None)
            for k, front in enumerate(fronts):
                cd_of_front = RankAndCrowdingSurvival.calc_crowding_distance(D['F'][front, :])
                #print(D['F'][front, :])
                _crowding[front] = cd_of_front
                _rank[front] = k+1

            crowding = D['crowding']

            is_equal = np.all(np.abs(_crowding - crowding) < 0.001)

            #if False:
            if not is_equal:
                print("-" * 30)
                print("Generation: ", i)
                print("Is rank equal: ", np.all(D['rank'] == _rank))
                index = np.where(np.abs(_crowding - crowding) > 0.001)
                print(index)
                print(D['rank'][index])
                print(D['F'][index])
                print(np.concatenate([_crowding[:, None], crowding[:, None]], axis=1)[index, :])

            #self.assertTrue(is_equal)


if __name__ == '__main__':
    unittest.main()
