import unittest

from pymoo.factory import get_reference_directions
from pymoo.util.reference_direction import sample_on_unit_simplex


class ReferenceDirectionsTest(unittest.TestCase):

    def test_das_dennis(self):
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        self.assertEqual(len(ref_dirs), 91)

    def test_das_dennis_achievable_points(self):
        ref_dirs = get_reference_directions("das-dennis", 3, n_points=91)
        self.assertEqual(len(ref_dirs), 91)

    def test_das_dennis_not_achievable_points(self):
        def fun():
            get_reference_directions("das-dennis", 3, n_points=92)
        self.assertRaises(Exception, fun)

    def test_unit_simplex_sampling(self):
        n_points = 1000
        n_dim = 3
        self.assertEqual(len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="das-dennis")), 990)
        self.assertEqual(len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="sum")), 1000)
        self.assertEqual(len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="kraemer")), 1000)


if __name__ == '__main__':
    unittest.main()
