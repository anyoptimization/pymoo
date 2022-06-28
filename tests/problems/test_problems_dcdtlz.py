import numpy as np
import pytest

from pymoo.core.individual import constr_to_cv, calc_cv
from pymoo.problems import get_problem
from tests.problems.test_correctness import load

PROBLEMS = ["dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3"]


@pytest.mark.parametrize('name', PROBLEMS)
def test_problems(name):
    problem = get_problem(name)

    X, F, CV = load("problems", name.upper(), attrs=["x", "f", "cv"])
    _F, _G = problem.evaluate(X, return_values_of=["F", "G"])
    _CV = calc_cv(G=_G)

    if _G.shape[1] > 1:
        # We need to do a special CV calculation to test for correctness since
        # the original code does not sum the violations but takes the maximum
        _CV = np.max(_G, axis=1)
        _CV = np.maximum(_CV, 0.0)

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(_CV, CV)


