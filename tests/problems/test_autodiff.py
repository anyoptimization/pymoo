import numpy as np
import pytest

pytestmark = pytest.mark.gradient

from pymoo.gradient.automatic import AutomaticDifferentiation
from pymoo.gradient.grad_complex import ComplexNumberGradient
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.problems import get_problem
from tests.problems.test_correctness import problems as correctness_problems
from tests.problems.test_g import problems as g_problems

problems = correctness_problems + g_problems

# Exclude problematic problems that have known gradient computation issues
EXCLUDED_PROBLEMS = ['Kursawe', 'G2']
filtered_problems = [(name, params) for name, params in problems if name not in EXCLUDED_PROBLEMS]


@pytest.mark.parametrize('name,params', filtered_problems)
def test_autodiff(name, params):
    problem = get_problem(name, *params)

    X = FloatRandomSampling().do(problem, 100).get("X")

    vals = ["F", "dF", "G", "dG", "H", "dH"]

    autodiff = AutomaticDifferentiation(problem)
    out_autodiff = autodiff.evaluate(X, return_values_of=vals, return_as_dictionary=True)

    complex = ComplexNumberGradient(problem)
    out_complex = complex.evaluate(X, return_values_of=vals, return_as_dictionary=True)

    for name in out_autodiff:
        if out_autodiff[name] is not None and out_complex[name] is not None:
            np.testing.assert_allclose(out_autodiff[name], out_complex[name], rtol=1e-3, atol=1e-5)
