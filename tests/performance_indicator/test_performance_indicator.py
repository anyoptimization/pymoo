import os
import unittest

import numpy as np

from pymoo.configuration import get_pymoo
from pymoo.factory import get_performance_indicator
from pymoo.performance_indicator.gd import GD
from pymoo.performance_indicator.igd import IGD
from tests.test_usage import test_usage


def get_indicators(pf):
    gd = get_performance_indicator("gd", pf)
    igd = get_performance_indicator("igd", pf)
    gd_plus = get_performance_indicator("gd+", pf)
    igd_plus = get_performance_indicator("igd+", pf)
    return gd, igd, gd_plus, igd_plus


class PerformanceIndicatorTest(unittest.TestCase):

    def test_usages(self):
        test_usage([os.path.join(get_pymoo(), "pymoo", "usage", "usage_performance_indicator.py")])

    # test whether they return the same as values from jmetalpy
    def test_values_of_indicators(self):
        l = [
            (GD, "gd"),
            (IGD, "igd")
        ]
        folder = os.path.join(get_pymoo(), "tests", "performance_indicator")
        pf = np.loadtxt(os.path.join(folder, "performance_indicators.pf"))

        for indicator, ext in l:

            for i in range(1, 5):
                F = np.loadtxt(os.path.join(folder, "performance_indicators_%s.f" % i))

                val = indicator(pf).calc(F)
                correct = np.loadtxt(os.path.join(folder, "performance_indicators_%s.%s" % (i, ext)))
                self.assertTrue(correct == val)

    def test_performance_indicator_1(self):
        A = np.array([2, 5])
        B = np.array([3, 9])
        D = np.array([2, 1])

        pf = np.array([[1, 0], [0, 10]])

        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(A), 5.099, places=2)
        self.assertAlmostEqual(gd.calc(B), 3.162, places=2)

        self.assertAlmostEqual(igd.calc(A), 5.242, places=2)
        self.assertAlmostEqual(igd.calc(B), 6.191, places=2)
        self.assertAlmostEqual(igd.calc(D), 5.32, places=2)

        self.assertAlmostEqual(igd_plus.calc(A), 3.550, places=2)
        self.assertAlmostEqual(igd_plus.calc(B), 6.110, places=2)

        self.assertAlmostEqual(gd_plus.calc(A), 2.0, places=2)
        self.assertAlmostEqual(gd_plus.calc(B), 3.0, places=2)

    def test_performance_indicator_2(self):
        A = np.array([5, 2])
        B = np.array([11, 3])
        pf = np.array([[0, 1], [10, 0]])
        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(A), 5.099, places=2)
        self.assertAlmostEqual(gd.calc(B), 3.162, places=2)

        self.assertAlmostEqual(gd_plus.calc(A), 2.0, places=2)
        self.assertAlmostEqual(gd_plus.calc(B), 3.162, places=2)

        self.assertAlmostEqual(igd.calc(A), 5.242, places=2)
        self.assertAlmostEqual(igd.calc(B), 7.171, places=2)

        self.assertAlmostEqual(igd_plus.calc(A), 3.550, places=2)
        self.assertAlmostEqual(igd_plus.calc(B), 7.171, places=2)

    def test_performance_indicator_3(self):
        A = np.array([2, 5])
        B = np.array([3, 9])
        C = np.array([10, 10])
        D = np.array([2, 1])
        pf = np.array([[1, 0], [0, 10]])
        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(D), 1.414, places=2)
        self.assertAlmostEqual(gd.calc(A), 5.099, places=2)

        self.assertAlmostEqual(gd_plus.calc(D), 1.414, places=2)
        self.assertAlmostEqual(gd_plus.calc(A), 2.00, places=2)

        self.assertAlmostEqual(igd.calc(D), 5.317, places=2)
        self.assertAlmostEqual(igd.calc(A), 5.242, places=2)

        self.assertAlmostEqual(igd_plus.calc(D), 1.707, places=2)
        self.assertAlmostEqual(igd_plus.calc(A), 3.550, places=2)

    def test_performance_indicator_4(self):
        A = np.array([[2, 4], [3, 3], [4, 2]])
        B = np.array([[2, 8], [4, 4], [8, 2]])
        pf = np.array([[0, 10], [1, 6], [2, 2], [6, 1], [10, 0]])
        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(A), 1.805, places=2)
        self.assertAlmostEqual(gd.calc(B), 2.434, places=2)

        self.assertAlmostEqual(gd_plus.calc(A), 1.138, places=2)
        self.assertAlmostEqual(gd_plus.calc(B), 2.276, places=2)

        self.assertAlmostEqual(igd.calc(A), 3.707, places=2)
        self.assertAlmostEqual(igd.calc(B), 2.591, places=2)

        self.assertAlmostEqual(igd_plus.calc(A), 1.483, places=2)
        self.assertAlmostEqual(igd_plus.calc(B), 2.260, places=2)

    def test_performance_indicator_5(self):
        A = np.array([[5, 2]])
        B = np.array([[6, 4], [10, 3]])
        pf = np.array([[0, 1], [10, 0]])
        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(A), 5.099, places=2)
        self.assertAlmostEqual(gd.calc(B), 4.328, places=2)

        self.assertAlmostEqual(gd_plus.calc(A), 2.0, places=2)
        self.assertAlmostEqual(gd_plus.calc(B), 3.5, places=2)

        self.assertAlmostEqual(igd.calc(A), 5.242, places=2)
        self.assertAlmostEqual(igd.calc(B), 4.854, places=2)

        self.assertAlmostEqual(igd_plus.calc(A), 3.550, places=2)
        self.assertAlmostEqual(igd_plus.calc(B), 4.854, places=2)

    def test_performance_indicator_6(self):
        A = np.array([[1, 5]])
        B = np.array([[5, 6]])
        pf = np.array([[4, 4]])
        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(A), 3.162, places=2)
        self.assertAlmostEqual(gd.calc(B), 2.236, places=2)

        self.assertAlmostEqual(gd_plus.calc(A), 1.0, places=2)
        self.assertAlmostEqual(gd_plus.calc(B), 2.236, places=2)

        self.assertAlmostEqual(igd.calc(A), 3.162, places=2)
        self.assertAlmostEqual(igd.calc(B), 2.236, places=2)

        self.assertAlmostEqual(igd_plus.calc(A), 1.0, places=2)
        self.assertAlmostEqual(igd_plus.calc(B), 2.236, places=2)

    def test_performance_indicator_8(self):
        A = np.array([[1, 8], [2, 2], [8, 1]])
        B = np.array([[4, 3]])
        pf = np.array([[0, 0]])
        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(A), 6.318, places=2)
        self.assertAlmostEqual(gd.calc(B), 5.0, places=2)

        self.assertAlmostEqual(gd_plus.calc(A), 6.318, places=2)
        self.assertAlmostEqual(gd_plus.calc(B), 5.0, places=2)

        self.assertAlmostEqual(igd.calc(A), 2.828, places=2)
        self.assertAlmostEqual(igd.calc(B), 5.0, places=2)

        self.assertAlmostEqual(igd_plus.calc(A), 2.828, places=2)
        self.assertAlmostEqual(igd_plus.calc(B), 5.0, places=2)

    def test_performance_indicator_9(self):
        A = np.array([[1, 8], [2, 2], [8, 1]])
        B = np.array([[2, 2]])
        pf = np.array([[0, 0]])
        gd, igd, gd_plus, igd_plus = get_indicators(pf)

        self.assertAlmostEqual(gd.calc(A), 6.318, places=2)
        self.assertAlmostEqual(gd.calc(B), 2.828, places=2)

        self.assertAlmostEqual(gd_plus.calc(A), 6.318, places=2)
        self.assertAlmostEqual(gd_plus.calc(B), 2.828, places=2)

        self.assertAlmostEqual(igd.calc(A), 2.828, places=2)
        self.assertAlmostEqual(igd.calc(B), 2.828, places=2)

        self.assertAlmostEqual(igd_plus.calc(A), 2.828, places=2)
        self.assertAlmostEqual(igd_plus.calc(B), 2.828, places=2)


if __name__ == '__main__':
    unittest.main()
