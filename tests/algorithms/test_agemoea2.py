import math
import unittest
import numpy as np

from pymoo.algorithms.moo.age2 import AGEMOEA2Survival
from pymoo.algorithms.moo.age2 import project_on_manifold
from pymoo.algorithms.moo.age2 import find_zero


class Test_AGEMOEA2Survival(unittest.TestCase):

    def test_project_on_manifold_p2(self):
        p = 2
        point = np.asarray([0.5, 0.5])
        projected_point = project_on_manifold(point, p)
        self.assertEqual(1 / np.power(2, 1 / p), projected_point[0], 0.001)

    def test_project_on_manifold_p1(self):
        p = 1
        point = np.asarray([0.5, 0.5])
        projected_point = project_on_manifold(point, p)
        self.assertEqual(0.5, projected_point[0], 0.001)

    def test_project_on_manifold_convex(self):
        p = 0.5
        point = np.asarray([0.5, 0.5])
        projected_point = project_on_manifold(point, p)
        self.assertAlmostEquals(0.25, projected_point[0], delta=0.0001)

    def test_compute_distance_p1(self):
        survival = AGEMOEA2Survival()
        p = 1

        front = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        matrix = survival.pairwise_distances(front, p)
        self.assertEqual(0, matrix[0][0])
        self.assertEqual(np.power(2, 0.5) / 2, matrix[0][1])
        self.assertEqual(np.power(2, 0.5), matrix[0][2])

    def test_compute_distance_p2(self):
        survival = AGEMOEA2Survival()
        p = 2

        front = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        matrix = survival.pairwise_distances(front, p)
        self.assertEqual(0, matrix[0][0])
        self.assertAlmostEquals(math.pi / 4, matrix[0][1], delta=0.05)
        self.assertAlmostEquals(math.pi / 2, matrix[0][2], delta=0.05)

    def test_find_zero_flat(self):
        survival = AGEMOEA2Survival()

        point = [0.5, 0.5]
        p = find_zero(point, 2, 0.001)
        self.assertEquals(1.0, p)

        point = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        p = find_zero(point, 3, 0.001)
        self.assertEquals(1.0, p)

    def test_find_zero_p2(self):
        survival = AGEMOEA2Survival()

        point = [1 / np.power(2, 0.5), 1 / np.power(2, 0.5)]
        p = find_zero(point, 2, 0.001)
        self.assertAlmostEquals(2.0, p, delta=0.0001)

    def test_compute_geometry_p1(self):
        survival = AGEMOEA2Survival()
        front = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        m, n = front.shape

        p = survival.compute_geometry(front, [0, 2], n)
        self.assertEqual(1, p)

    def test_compute_geometry_p2(self):
        survival = AGEMOEA2Survival()
        mid_point = [1 / np.power(2, 0.5), 1 / np.power(2, 0.5)]
        front = np.array([[1.0, 0.0], mid_point, [0.0, 1.0]])
        m, n = front.shape

        p = survival.compute_geometry(front, [0, 2], n)
        self.assertAlmostEquals(2.0, p, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
