import numpy as np
import pytest

from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.problems import get_problem
from tests.problems.test_correctness import load

PROBLEMS = [f"mw{k}" for k in range(1, 15)]


@pytest.mark.parametrize('name', PROBLEMS)
def test_mw(name):
    problem = get_problem(name)

    X, F, CV = load(name.upper(), suffix=["MW"])
    _F, _G = problem.evaluate(X, return_values_of=["F", "G"])
    _CV = TotalConstraintViolation(aggr_func=np.sum).calc(_G)

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(_CV, CV)
