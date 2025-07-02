import numpy as np
import pytest

from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.indicators.hv.exact_2d import ExactHypervolume2D
from pymoo.indicators.hv.approximate import ApproximateHypervolume
from pymoo.problems.many import DTLZ1
from pymoo.problems.multi import ZDT1


def case_2d():
    np.random.seed(1)

    ref_point = np.array([1.5, 1.5])

    F = ZDT1().pareto_front()
    F = F[::10] * 1.2
    F = F[np.random.permutation(len(F))]

    return ref_point, F


def case_2d_smaller_ref():
    _, F = case_2d()
    ref_point = np.array([0.9, 0.9])
    return ref_point, F


def case_3d():
    np.random.seed(1)

    ref_point = np.array([1.5, 1.5, 1.5])

    F = DTLZ1().pareto_front()
    F = F[::10] * 1.2
    F = F[np.random.permutation(len(F))]

    return ref_point, F


def test_hvc_2d():
    np.random.seed(1)
    ref_point, F = case_2d()

    exact = ExactHypervolume(ref_point).add(F)
    exact2d = ExactHypervolume2D(ref_point).add(F)

    assert np.allclose(exact.hv, exact2d.hv)
    np.testing.assert_allclose(exact.hvc, exact2d.hvc)

    for i in range(len(F)):
        k = np.random.randint(low=0, high=len(F) - i)

        exact.delete(k)
        exact2d.delete(k)

        assert np.allclose(exact.hv, exact2d.hv)
        np.testing.assert_allclose(exact.hvc, exact2d.hvc)


@pytest.mark.parametrize('case', [case_2d(), case_3d()])
def test_hvc_monte_carlo(case):
    # Set a fixed seed for the entire test to ensure deterministic behavior
    np.random.seed(1)
    random_state = np.random.RandomState(1)
    ref_point, F = case

    exact = ExactHypervolume(ref_point).add(F)
    mc = ApproximateHypervolume(ref_point, F=F, n_samples=20_000)

    assert np.allclose(exact.hv, mc.hv, rtol=0, atol=1.5e-2)
    np.testing.assert_allclose(exact.hvc, mc.hvc, rtol=0, atol=1.5e-1)

    for i in range(len(F)):
        k = random_state.randint(low=0, high=len(F) - i)

        exact.delete(k)
        mc.delete(k)

        assert np.allclose(exact.hv, mc.hv, rtol=0, atol=1.5e-2)
        np.testing.assert_allclose(exact.hvc, mc.hvc, rtol=0, atol=1.5e-1)
