import numpy as np
import pytest

from pymoo.core.problem import ieq_cv
from pymoo.factory import get_problem
from tests.problems.test_correctness import load

PROBLEMS = [f"mw{k}" for k in range(1, 15)]


@pytest.mark.parametrize('name', PROBLEMS)
def test_mw(name):
    problem = get_problem(name)

    X, F, CV = load(name.upper(), suffix=["MW"])
    _F, _G = problem.evaluate(X, return_values_of=["F", "G"])
    _CV = ieq_cv(_G)[:, 0]

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(_CV, CV)
