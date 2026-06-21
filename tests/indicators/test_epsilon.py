import numpy as np
import pytest
import moocore

from pymoo.indicators.epsilon import Epsilon, EpsilonMultiplicative
from pymoo.problems.multi import ZDT1, ZDT2, ZDT3


@pytest.fixture(params=[ZDT1, ZDT2, ZDT3])
def epsilon_case(request):
    np.random.seed(1)
    pf = request.param().pareto_front() + 1.0   # shift away from 0 for epsilon_mult
    F = pf + np.random.uniform(0, 0.1, pf.shape)
    return pf, F


def test_epsilon_additive(epsilon_case):
    pf, F = epsilon_case
    assert np.isclose(Epsilon(pf).do(F), moocore.epsilon_additive(F, ref=pf))


def test_epsilon_multiplicative(epsilon_case):
    pf, F = epsilon_case
    assert np.isclose(EpsilonMultiplicative(pf).do(F), moocore.epsilon_mult(F, ref=pf))
