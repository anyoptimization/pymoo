import numpy as np
import pytest

from tests.gradients.grad_problem import ZDT1WithGradient, ZDT2WithGradient, ZDT3WithGradient, \
    AutomaticDifferentiationProblem, SphereWithGradientAndConstraint

from pymoo.experimental.numdiff import NumericalDifferentiation, OneSidedJacobian, CentralJacobian, ComplexStepJacobian
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.problems.multi.zdt import ZDT1, ZDT2, ZDT3


class ElementwiseZDT1(Problem):

    def __init__(self, n_var=30, n_obj=2, n_constr=0, **kwargs):
        super().__init__(n_var, evaluation_of=["F", "dF"], elementwise_evaluation=True, n_obj=n_obj, n_constr=n_constr,
                         **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]
        g = 1 + 9.0 / (self.n_var - 1) * x[1:].sum()
        f2 = g * (1 - (f1 / g) ** 0.5)
        out["F"] = ([f1, f2])


@pytest.mark.skip()
def test_automatic_differentiation_gradient():
    for entry in [(ZDT1(), ZDT1WithGradient()),
                  (ZDT2(), ZDT2WithGradient()),
                  (ZDT3(), ZDT3WithGradient())]:
        auto_diff, correct, = entry

        X = np.random.random((100, correct.n_var))

        F, dF = correct.evaluate(X, return_values_of=["F", "dF"])
        _F, _dF = auto_diff.evaluate(X, return_values_of=["F", "dF"])

        np.testing.assert_allclose(_F, F, rtol=1e-5, atol=0)
        np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)


@pytest.mark.skip()
def test_numerical_differentiation_gradient():
    for entry in [(ZDT1(), ZDT1WithGradient()),
                  (ZDT2(), ZDT2WithGradient()),
                  (ZDT3(), ZDT3WithGradient()),
                  # (MySphere(), MySphereWithGradient())
                  ]:
        problem, correct, = entry

        np.random.seed(1)
        X = np.random.random((100, correct.n_var))

        F, dF = correct.evaluate(X, return_values_of=["F", "dF"])

        _dF = NumericalDifferentiation(jacobian=OneSidedJacobian()).do(problem, X, F)
        np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)

        _dF = NumericalDifferentiation(jacobian=CentralJacobian()).do(problem, X, F)
        np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)

        _dF = NumericalDifferentiation(jacobian=ComplexStepJacobian()).do(problem, X, F)
        np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)

        # _dF = ForwardNumericalDifferentiation().do(problem, Solution(X=X[0], F=F[0]))
        # _dF = ForwardNumericalDifferentiation().do(problem, Population.new(X=X, F=F))


@pytest.mark.skip()
def test_numerical_differentiation_hessian():
    np.random.seed(1)

    rosen = lambda x: (1 - x[0]) ** 2 + 105. * (x[1] - x[0] ** 2) ** 2
    cubic = lambda x: (x ** 3).sum()
    sphere = lambda x: (x ** 2).sum()
    for func in [
        sphere,
        cubic,
        rosen,
    ]:
        problem = AutomaticDifferentiationProblem(func, n_var=5)

        X = np.random.random((1, problem.n_var))
        F, dF, ddF = problem.evaluate(X, return_values_of=["F", "dF", "ddF"])

        evaluator = Evaluator()
        _dF, _ddF = NumericalDifferentiation(eps=None).do(problem, X, F, evaluator=evaluator, hessian=True)
        print(evaluator.n_eval)
        np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)
        np.testing.assert_allclose(_ddF, ddF, rtol=1e-5, atol=0)
        print(func)

    print("Done")


@pytest.mark.skip()
def test_numerical_differentiation_with_gradient():
    np.random.seed(1)

    problem = SphereWithGradientAndConstraint()

    X = np.random.random((1, problem.n_var))
    F, G, dF, ddF, dG, ddG = problem.evaluate(X, return_values_of=["F", "G", "dF", "ddF", "dG", "ddG"])

    evaluator = Evaluator()
    _dF, _ddF, _dG, _ddG = NumericalDifferentiation(eps=None).do(problem, X, F, G=G, evaluator=evaluator, hessian=True)

    print(evaluator.n_eval)
    np.testing.assert_allclose(_dF, dF, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_ddF, ddF, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_dG, dG, rtol=1e-5, atol=0)
    np.testing.assert_allclose(_ddG, ddG, rtol=1e-5, atol=0)

    print("Done")
