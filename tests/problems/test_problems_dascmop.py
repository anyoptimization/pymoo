import numpy as np
import pytest

from pymoo.core.individual import calc_cv
from pymoo.problems import get_problem
from pymoo.problems.multi import DIFFICULTIES
from tests.problems.test_correctness import load


@pytest.mark.parametrize("i", range(1, 10))
@pytest.mark.parametrize("j", range(len(DIFFICULTIES)))
def test_dascomp(i, j):
    name, difficulty = f"dascmop{i}", DIFFICULTIES[j]
    problem = get_problem(name, difficulty)

    X, F, CV = load("problems", "DASCMOP", str(j), name.upper(), attrs=["x", "f", "cv"])
    _F, _G = problem.evaluate(X, return_values_of=["F", "G"])
    _CV = calc_cv(G=_G)

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(-_CV, CV)

