import unittest

import numpy as np

from pymoo.tests.test_util import make_individual
from pymoo.util.dominator import Dominator


class DominationTest(unittest.TestCase):
    def test_first_has_less_constraint_violation(self):
        i1 = make_individual(np.array([]), np.array([0.0]))
        i2 = make_individual(np.array([]), np.array([1.0]))
        self.assertEqual(1, Dominator.get_relation(i1, i2))
        self.assertEqual(True, Dominator.is_dominating(i1, i2))

    def test_second_has_less_constraint_violation(self):
        i1 = make_individual(np.array([]), np.array([1.0]))
        i2 = make_individual(np.array([]), np.array([0.0]))
        self.assertEqual(-1, Dominator.get_relation(i1, i2))
        self.assertEqual(False, Dominator.is_dominating(i1, i2))

    def test_equal_constraint_violation(self):
        i1 = make_individual(np.array([]), np.array([1.0]))
        i2 = make_individual(np.array([]), np.array([1.0]))
        self.assertEqual(0, Dominator.get_relation(i1, i2))
        self.assertEqual(False, Dominator.is_dominating(i1, i2))

    def test_first_is_dominating(self):
        i1 = make_individual(np.array([0.0, 0.0]), np.array([1.0]))
        i2 = make_individual(np.array([1.0, 0.0]), np.array([1.0]))
        self.assertEqual(1, Dominator.get_relation(i1, i2))
        self.assertEqual(True, Dominator.is_dominating(i1, i2))

    def test_equal(self):
        i1 = make_individual(np.array([0.0, 0.0]), np.array([1.0]))
        i2 = make_individual(np.array([0.0, 0.0]), np.array([1.0]))
        self.assertEqual(0, Dominator.get_relation(i1, i2))
        self.assertEqual(False, Dominator.is_dominating(i1, i2))

    def test_indifferent(self):
        i1 = make_individual(np.array([1.0, 0.0]), np.array([1.0]))
        i2 = make_individual(np.array([0.0, 2.0]), np.array([1.0]))
        self.assertEqual(0, Dominator.get_relation(i1, i2))
        self.assertEqual(False, Dominator.is_dominating(i1, i2))



if __name__ == '__main__':
    unittest.main()
