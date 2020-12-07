import unittest

import numpy as np

from pymoo.algorithms.nsga2 import calc_crowding_distance


class CrowdingDistanceTest(unittest.TestCase):

    """
    def test_crowding_distance(self):
        D = np.loadtxt(os.path.join(get_pymoo(), "tests", "resources", "test_crowding.dat"))
        F, cd = D[:, :-1], D[:, -1]
        self.assertTrue(np.all(np.abs(cd - calc_crowding_distance_vectorized(F)) < 0.001))
    """

    def test_crowding_distance_one_duplicate(self):
        F = np.array([[1.0, 1.0], [1.0, 1.0], [0.5, 1.5], [0.0, 2.0]])
        cd = calc_crowding_distance(F)
        self.assertTrue(np.all(np.isclose(cd, np.array([np.inf, 0.0, 1.0, np.inf]))))

    def test_crowding_distance_two_duplicates(self):
        F = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.5, 1.5], [0.0, 2.0]])
        cd = calc_crowding_distance(F)
        self.assertTrue(np.all(np.isclose(cd, np.array([np.inf, 0.0, 0.0, 1.0, np.inf]))))

    def test_crowding_distance_norm_equals_zero(self):
        F = np.array([[1.0, 1.5, 0.5, 1.0], [1.0, 0.5, 1.5, 1.0], [1.0, 0.0, 2.0, 1.5]])
        cd = calc_crowding_distance(F)
        self.assertTrue(np.all(np.isclose(cd, np.array([np.inf, 0.75, np.inf]))))


if __name__ == '__main__':
    unittest.main()
