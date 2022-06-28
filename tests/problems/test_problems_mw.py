import numpy as np
import pytest

from pymoo.core.individual import constr_to_cv, calc_cv
from pymoo.problems import get_problem
from tests.problems.test_correctness import load

PROBLEMS = [f"mw{k}" for k in range(1, 15)]


@pytest.mark.parametrize('name', PROBLEMS)
def test_mw(name):
    problem = get_problem(name)

    X, F, CV = load("problems", "MW", name.upper(), attrs=["x", "f", "cv"])

    _F, _G = problem.evaluate(X, return_values_of=["F", "G"])
    _CV = calc_cv(_G)

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(_CV, CV)
