import numpy as np
import pytest

from pymoo.factory import get_problem
from tests.problems.test_correctness import load

PROBLEMS = ["dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3"]


@pytest.mark.parametrize('name', PROBLEMS)
def test_problems(name):
    problem = get_problem(name)

    X, F, CV = load(name.upper())
    _F, _CV, _G = problem.evaluate(X, return_values_of=["F", "CV", "G"])

    if _G.shape[1] > 1:
        # We need to do a special CV calculation to test for correctness since
        # the original code does not sum the violations but takes the maximum
        _CV = np.max(_G, axis=1)[:, None]
        _CV = np.maximum(_CV, 0)

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(_CV[:, 0], CV)


