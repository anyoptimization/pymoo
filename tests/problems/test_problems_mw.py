import numpy as np
import pytest

from pymoo.factory import get_problem
from tests.problems.test_correctness import load

PROBLEMS = [f"mw{k}" for k in range(1, 15)]


@pytest.mark.parametrize('name', PROBLEMS)
def test_mw(name):
    problem = get_problem(name)

    X, F, CV = load(name.upper(), suffix=["MW"])
    _F, _CV = problem.evaluate(X, return_values_of=["F", "CV"])

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(_CV[:, 0], CV)
