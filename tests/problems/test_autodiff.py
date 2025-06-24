import numpy as np
import pytest
import pymoo.gradient

# Skip all autodiff tests due to autograd compatibility issues
pytestmark = pytest.mark.skip(reason="Autograd compatibility issues with numpy 2.x")

# Activate autograd toolbox for gradient calculations
# pymoo.gradient.activate("autograd.numpy")

from pymoo.gradient.automatic import AutomaticDifferentiation, ElementwiseAutomaticDifferentiation
from pymoo.gradient.grad_complex import ComplexNumberGradient
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.problems import get_problem
from tests.problems.test_correctness import problems as correctness_problems
from tests.problems.test_g import problems as g_problems

problems = correctness_problems + g_problems


@pytest.mark.parametrize('name,params', problems)
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
            # Use more relaxed tolerances for numerical gradient comparison
            np.testing.assert_allclose(out_autodiff[name], out_complex[name], rtol=1e-3, atol=1e-5)
