import numpy as np
import pytest

from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.problems import get_problem
from pymoo.problems.multi import DIFFICULTIES
from tests.problems.test_correctness import load


@pytest.mark.parametrize("i", range(1, 10))
@pytest.mark.parametrize("j", range(len(DIFFICULTIES)))
def test_dascomp(i, j):
    name, difficulty = f"dascmop{i}", DIFFICULTIES[j]
    problem = get_problem(name, difficulty)

    X, F, CV = load(name.upper(), suffix=["DASCMOP", str(j)])
    _F, _G = problem.evaluate(X, return_values_of=["F", "G"])
    _CV = TotalConstraintViolation(aggr_func=np.sum).calc(_G)

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(-_CV, CV)
