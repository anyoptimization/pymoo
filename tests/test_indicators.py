import numpy as np

from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from tests.test_util import load_to_test_resource


def test_values_of_indicators():
    l = [
        (GD, "gd"),
        (IGD, "igd")
    ]

    pf = load_to_test_resource("performance_indicator", f"performance_indicators.pf", to="numpy")

    for indicator, ext in l:

        for i in range(1, 5):

            F = load_to_test_resource("performance_indicator", f"performance_indicators_{i}.f", to="numpy")
            val = indicator(pf).do(F)

            correct = load_to_test_resource("performance_indicator", f"performance_indicators_{i}.{ext}", to="numpy")
            assert float(correct) == val


def test_performance_indicator_1():
    A = np.array([2, 5])
    B = np.array([3, 9])
    D = np.array([2, 1])

    pf = np.array([[1, 0], [0, 10]])

    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(A), 5.099, decimal=2)
    np.testing.assert_almost_equal(gd.do(B), 3.162, decimal=2)

    np.testing.assert_almost_equal(igd.do(A), 5.242, decimal=2)
    np.testing.assert_almost_equal(igd.do(B), 6.191, decimal=2)
    np.testing.assert_almost_equal(igd.do(D), 5.32, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(A), 3.550, decimal=2)
    np.testing.assert_almost_equal(igd_plus.do(B), 6.110, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(A), 2.0, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(B), 3.0, decimal=2)


def test_performance_indicator_2():
    A = np.array([5, 2])
    B = np.array([11, 3])
    pf = np.array([[0, 1], [10, 0]])
    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(A), 5.099, decimal=2)
    np.testing.assert_almost_equal(gd.do(B), 3.162, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(A), 2.0, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(B), 3.162, decimal=2)

    np.testing.assert_almost_equal(igd.do(A), 5.242, decimal=2)
    np.testing.assert_almost_equal(igd.do(B), 7.171, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(A), 3.550, decimal=2)
    np.testing.assert_almost_equal(igd_plus.do(B), 7.171, decimal=2)


def test_performance_indicator_3():
    A = np.array([2, 5])
    B = np.array([3, 9])
    C = np.array([10, 10])
    D = np.array([2, 1])
    pf = np.array([[1, 0], [0, 10]])
    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(D), 1.414, decimal=2)
    np.testing.assert_almost_equal(gd.do(A), 5.099, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(D), 1.414, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(A), 2.00, decimal=2)

    np.testing.assert_almost_equal(igd.do(D), 5.317, decimal=2)
    np.testing.assert_almost_equal(igd.do(A), 5.242, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(D), 1.707, decimal=2)
    np.testing.assert_almost_equal(igd_plus.do(A), 3.550, decimal=2)


def test_performance_indicator_4():
    A = np.array([[2, 4], [3, 3], [4, 2]])
    B = np.array([[2, 8], [4, 4], [8, 2]])
    pf = np.array([[0, 10], [1, 6], [2, 2], [6, 1], [10, 0]])
    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(A), 1.805, decimal=2)
    np.testing.assert_almost_equal(gd.do(B), 2.434, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(A), 1.138, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(B), 2.276, decimal=2)

    np.testing.assert_almost_equal(igd.do(A), 3.707, decimal=2)
    np.testing.assert_almost_equal(igd.do(B), 2.591, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(A), 1.483, decimal=2)
    np.testing.assert_almost_equal(igd_plus.do(B), 2.260, decimal=2)


def test_performance_indicator_5():
    A = np.array([[5, 2]])
    B = np.array([[6, 4], [10, 3]])
    pf = np.array([[0, 1], [10, 0]])
    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(A), 5.099, decimal=2)
    np.testing.assert_almost_equal(gd.do(B), 4.328, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(A), 2.0, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(B), 3.5, decimal=2)

    np.testing.assert_almost_equal(igd.do(A), 5.242, decimal=2)
    np.testing.assert_almost_equal(igd.do(B), 4.854, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(A), 3.550, decimal=2)
    np.testing.assert_almost_equal(igd_plus.do(B), 4.854, decimal=2)


def test_performance_indicator_6():
    A = np.array([[1, 5]])
    B = np.array([[5, 6]])
    pf = np.array([[4, 4]])
    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(A), 3.162, decimal=2)
    np.testing.assert_almost_equal(gd.do(B), 2.236, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(A), 1.0, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(B), 2.236, decimal=2)

    np.testing.assert_almost_equal(igd.do(A), 3.162, decimal=2)
    np.testing.assert_almost_equal(igd.do(B), 2.236, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(A), 1.0, decimal=2)
    np.testing.assert_almost_equal(igd_plus.do(B), 2.236, decimal=2)


def test_performance_indicator_8():
    A = np.array([[1, 8], [2, 2], [8, 1]])
    B = np.array([[4, 3]])
    pf = np.array([[0, 0]])
    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(A), 6.318, decimal=2)
    np.testing.assert_almost_equal(gd.do(B), 5.0, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(A), 6.318, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(B), 5.0, decimal=2)

    np.testing.assert_almost_equal(igd.do(A), 2.828, decimal=2)
    np.testing.assert_almost_equal(igd.do(B), 5.0, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(A), 2.828, decimal=2)
    np.testing.assert_almost_equal(igd_plus.do(B), 5.0, decimal=2)


def test_performance_indicator_9():
    A = np.array([[1, 8], [2, 2], [8, 1]])
    B = np.array([[2, 2]])
    pf = np.array([[0, 0]])
    gd, igd, gd_plus, igd_plus = get_indicators(pf)

    np.testing.assert_almost_equal(gd.do(A), 6.318, decimal=2)
    np.testing.assert_almost_equal(gd.do(B), 2.828, decimal=2)

    np.testing.assert_almost_equal(gd_plus.do(A), 6.318, decimal=2)
    np.testing.assert_almost_equal(gd_plus.do(B), 2.828, decimal=2)

    np.testing.assert_almost_equal(igd.do(A), 2.828, decimal=2)
    np.testing.assert_almost_equal(igd.do(B), 2.828, decimal=2)

    np.testing.assert_almost_equal(igd_plus.do(A), 2.828, decimal=3)
    np.testing.assert_almost_equal(igd_plus.do(B), 2.828, decimal=3)




def get_indicators(pf):
    return tuple([clazz(pf) for clazz in [GD, IGD, GDPlus, IGDPlus]])
