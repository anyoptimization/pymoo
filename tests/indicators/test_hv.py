import numpy as np
import pytest
import moocore

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


def case_3d():
    np.random.seed(1)
    ref_point = np.array([1.5, 1.5, 1.5])
    F = DTLZ1().pareto_front()
    F = F[::10] * 1.2
    F = F[np.random.permutation(len(F))]
    return ref_point, F


@pytest.mark.parametrize('case', [case_2d(), case_3d()])
def test_hvc_approximate(case):
    np.random.seed(1)
    ref_point, F = case

    exact_hv = moocore.hypervolume(F, ref=ref_point)
    approx = ApproximateHypervolume(ref_point).add(F)

    assert np.isclose(exact_hv, approx.hv, rtol=0, atol=1e-3)
    np.testing.assert_array_equal(approx.hvc, moocore.hv_contributions(F, ref=ref_point))

    while len(F) > 0:
        k = np.random.randint(low=0, high=len(F))
        F = np.delete(F, k, axis=0)
        approx.delete(k)

        exact_hv = moocore.hypervolume(F, ref=ref_point) if len(F) > 0 else 0.0
        assert np.isclose(exact_hv, approx.hv, rtol=0, atol=1e-3)
