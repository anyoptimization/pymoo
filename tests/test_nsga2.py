import os
import pickle
import unittest

import numpy as np

from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival, calc_crowding_distance
from pymoo.rand.impl.custom_random_generator import CustomRandomGenerator
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class NSGA2Test(unittest.TestCase):
    """
    Some methods compare with one test run from the C code (200 generations) and
    tests if the rank is always equal.

    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join("resources", "cnsga2_run_zdt4.dat"), 'rb') as f:
            cls.data = pickle.load(f)

    # tests whether the first number by the random generator is equal
    def test_custom_random_generator(self):
        rand = CustomRandomGenerator(0.1)
        val = rand.random()
        self.assertAlmostEqual(val, 0.337237, places=6)

    def test_rank_and_crowding_distance(self):
        for i, D in enumerate(self.data):

            survivor_and_last_front = np.where(D['rank'] != -1.0)[0]
            crowding = D['crowding'][survivor_and_last_front]
            rank = D['rank'][survivor_and_last_front].astype(np.int)
            F = D['F'][survivor_and_last_front, :]

            _rank = np.full(rank.shape, -1)
            _crowding = np.full(crowding.shape, -1.0)
            fronts = NonDominatedSorting().do(F)
            for k, front in enumerate(fronts):
                cd_of_front = calc_crowding_distance(F[front, :])
                _crowding[front] = cd_of_front
                _rank[front] = k + 1

            is_equal = np.all(rank == _rank)
            if not is_equal:
                index = np.where(rank == _rank)
                print(index)
                print(D['rank'][index])
                print(D['F'][index])

            self.assertTrue(is_equal)

            is_equal = np.all(np.abs(_crowding - crowding) < 0.001)
            if not is_equal:

                index = np.where(np.abs(_crowding - crowding) > 0.001)[0]
                index = index[np.argsort(rank[index])]

                # only an error if it is not a duplicate F value
                for i_not_equal in index:

                    if len(np.where(np.all(F[i_not_equal, :] == F, axis=1))[0]) == 1:
                        print("-" * 30)
                        print("Generation: ", i)
                        print("Is rank equal: ", np.all(rank == _rank))

                        print(index)
                        print(rank[index])
                        print(F[index])
                        print(np.concatenate([_crowding[:, None], crowding[:, None]], axis=1)[index, :])
                        print()

                        self.assertTrue(is_equal)


if __name__ == '__main__':
    unittest.main()
