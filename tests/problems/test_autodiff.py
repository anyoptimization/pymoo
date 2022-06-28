import pytest

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.problems import get_problem
from pymoo.gradient.automatic import AutomaticDifferentiation
from tests.problems.test_correctness import problems as correctness_problems
from tests.problems.test_g import problems as g_problems

problems = correctness_problems + g_problems


@pytest.mark.parametrize('name,params', problems)
def test_autodiff(name, params):
    problem = AutomaticDifferentiation(get_problem(name, *params))
    X = FloatRandomSampling().do(problem, 100).get("X")
    out = problem.evaluate(X, return_values_of=["F", "dF", "G", "dG", "H", "dH"])
    assert out is not None
