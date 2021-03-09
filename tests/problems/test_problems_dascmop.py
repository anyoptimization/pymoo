import numpy as np
import pytest

from pymoo.factory import get_problem
from pymoo.problems.multi import DIFFICULTIES
from tests.problems.test_correctness import load


@pytest.mark.parametrize("i", range(1, 10))
@pytest.mark.parametrize("j", range(len(DIFFICULTIES)))
def test_dascomp(i, j):
    name, difficulty = f"dascmop{i}", DIFFICULTIES[j]
    problem = get_problem(name, difficulty)

    X, F, CV = load(name.upper(), suffix=["DASCMOP", str(j)])
    _F, _CV = problem.evaluate(X, return_values_of=["F", "CV"])

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(-_CV[:, 0], CV)
