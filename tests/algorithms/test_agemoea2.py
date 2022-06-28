import math

import numpy as np

from pymoo.algorithms.moo.age2 import AGEMOEA2Survival
from pymoo.algorithms.moo.age2 import find_zero
from pymoo.algorithms.moo.age2 import project_on_manifold


def test_project_on_manifold_p2():
    p = 2
    point = np.asarray([0.5, 0.5])
    projected_point = project_on_manifold(point, p)

    np.testing.assert_almost_equal(1 / np.power(2, 1 / p), projected_point[0])


def test_project_on_manifold_p1():
    p = 1
    point = np.asarray([0.5, 0.5])
    projected_point = project_on_manifold(point, p)
    np.testing.assert_almost_equal(0.5, projected_point[0])


def test_project_on_manifold_convex():
    p = 0.5
    point = np.asarray([0.5, 0.5])
    projected_point = project_on_manifold(point, p)

    np.testing.assert_almost_equal(0.25, projected_point[0])


def test_compute_distance_p1():
    survival = AGEMOEA2Survival()
    p = 1

    front = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    matrix = survival.pairwise_distances(front, p)
    np.testing.assert_almost_equal(0, matrix[0][0])
    np.testing.assert_almost_equal(np.power(2, 0.5) / 2, matrix[0][1])
    np.testing.assert_almost_equal(np.power(2, 0.5), matrix[0][2])


def test_compute_distance_p2():
    survival = AGEMOEA2Survival()
    p = 2

    front = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    matrix = survival.pairwise_distances(front, p)

    np.testing.assert_almost_equal(0, matrix[0][0], decimal=3)
    np.testing.assert_almost_equal(math.pi / 4, matrix[0][1], decimal=2)
    np.testing.assert_almost_equal(math.pi / 2, matrix[0][2], decimal=1)


def test_find_zero_flat():
    point = [0.5, 0.5]
    p = find_zero(point, 2, 0.001)
    np.testing.assert_almost_equal(1.0, p)

    point = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    p = find_zero(point, 3, 0.001)

    np.testing.assert_almost_equal(1.0, p)


def test_find_zero_p2():
    point = [1 / np.power(2, 0.5), 1 / np.power(2, 0.5)]
    p = find_zero(point, 2, 0.001)
    np.testing.assert_almost_equal(2.0, p)


def test_compute_geometry_p1():
    survival = AGEMOEA2Survival()
    front = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    m, n = front.shape

    p = survival.compute_geometry(front, [0, 2], n)
    np.testing.assert_almost_equal(1, p)


def test_compute_geometry_p2():
    survival = AGEMOEA2Survival()
    mid_point = [1 / np.power(2, 0.5), 1 / np.power(2, 0.5)]
    front = np.array([[1.0, 0.0], mid_point, [0.0, 1.0]])
    m, n = front.shape

    p = survival.compute_geometry(front, [0, 2], n)
    np.testing.assert_almost_equal(2.0, p)
