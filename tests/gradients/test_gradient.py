import numpy as np
import pytest

from pymoo.gradient.automatic import ElementwiseAutomaticDifferentiation, AutomaticDifferentiation
from pymoo.problems.multi.zdt import ZDT1
from tests.gradients.problems_with_gradients import MySphereWithGradient, MySphere, ZDT1WithGradient, ElementwiseZDT1, \
    MyConstrainedSphereWithGradient, MyConstrainedSphere, ConstrainedZDT1WithGradient, ConstrainedZDT1

# Mark all gradient tests with gradient marker
pytestmark = pytest.mark.gradient


@pytest.mark.parametrize("correct, problem", [
    (MySphereWithGradient(), MySphere()),
    pytest.param(ZDT1WithGradient(), ElementwiseZDT1())], ids=['elementwise_sphere', 'elementwise_zdt1'])
def test_elementwise_eval_with_gradient(correct, problem):
    X = np.random.random((100, correct.n_var))

    autodiff = ElementwiseAutomaticDifferentiation(problem)

    F, dF = correct.evaluate(X, return_values_of=["F", "dF"])
    _F, _dF = autodiff.evaluate(X, return_values_of=["F", "dF"])

    np.testing.assert_allclose(_F, F, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)


@pytest.mark.parametrize("correct, problem", [(MyConstrainedSphereWithGradient(), MyConstrainedSphere())],
                         ids=['elementwise_constr_sphere'])
def test_vectorized_constrained_eval_with_gradient(correct, problem):
    X = np.random.random((100, correct.n_var))

    autodiff = AutomaticDifferentiation(problem)

    F, dF, G, dG = correct.evaluate(X, return_values_of=["F", "dF", "G", "dG"])
    _F, _dF, _G, _dG = autodiff.evaluate(X, return_values_of=["F", "dF", "G", "dG"])

    np.testing.assert_allclose(_F, F, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)

    np.testing.assert_allclose(_G, G, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_dG, dG, rtol=1e-5, atol=0)


@pytest.mark.parametrize("correct, problem", [
    (ZDT1WithGradient(), ZDT1())], ids=['vectorized_zdt1'])
def test_vectorized_eval_with_gradient(correct, problem):
    X = np.random.random((100, correct.n_var))

    autodiff = AutomaticDifferentiation(problem)

    F, dF = correct.evaluate(X, return_values_of=["F", "dF"])
    _F, _dF = autodiff.evaluate(X, return_values_of=["F", "dF"])

    np.testing.assert_almost_equal(F, _F)
    np.testing.assert_almost_equal(_dF, dF, decimal=5)


@pytest.mark.parametrize("correct, problem", [(ConstrainedZDT1WithGradient(), ConstrainedZDT1())],
                         ids=['vectorized_constr_zdt1'])
def test_constrained_multi_eval_with_gradient(correct, problem):
    X = np.random.random((100, correct.n_var))

    autodiff = AutomaticDifferentiation(problem)

    F, dF, G, dG = correct.evaluate(X, return_values_of=["F", "dF", "G", "dG"])
    _F, _dF, _G, _dG = autodiff.evaluate(X, return_values_of=["F", "dF", "G", "dG"])

    np.testing.assert_allclose(_F, F, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)

    np.testing.assert_allclose(_G, G, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_dG, dG, rtol=1e-5, atol=0)




