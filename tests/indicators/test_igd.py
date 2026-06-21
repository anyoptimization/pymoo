import numpy as np
import pytest
import moocore

from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems.multi import ZDT1, ZDT2, ZDT3


@pytest.fixture(params=[ZDT1, ZDT2, ZDT3])
def igd_case(request):
    np.random.seed(1)
    pf = request.param().pareto_front()
    F = pf + np.random.uniform(0, 0.1, pf.shape)
    return pf, F


def test_igd_matches_moocore(igd_case):
    pf, F = igd_case
    assert np.isclose(IGD(pf).do(F), moocore.igd(F, ref=pf))


def test_igd_plus_matches_moocore(igd_case):
    pf, F = igd_case
    assert np.isclose(IGDPlus(pf).do(F), moocore.igd_plus(F, ref=pf))
