import unittest

import numpy as np

from pymoo.model.evaluator import Evaluator
from pymoo.model.problem import Problem
from pymoo.problems.multi.zdt import ZDT1, ZDT2, ZDT3
from pymoo.util.differentiation.numerical import NumericalDifferentiation, OneSidedJacobian, CentralJacobian, \
    ComplexStepJacobian
from tests.gradients.grad_problem import ZDT2WithGradient, ZDT3WithGradient, ZDT1WithGradient, \
    AutomaticDifferentiationProblem, SphereWithGradientAndConstraint


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


class GradientTest(unittest.TestCase):

    def test_automatic_differentiation_gradient(self):
        for entry in [(ZDT1(), ZDT1WithGradient()),
                      (ZDT2(), ZDT2WithGradient()),
                      (ZDT3(), ZDT3WithGradient())]:
            auto_diff, correct, = entry

            X = np.random.random((100, correct.n_var))

            F, dF = correct.evaluate(X, return_values_of=["F", "dF"])
            _F, _dF = auto_diff.evaluate(X, return_values_of=["F", "dF"])

            self.assertTrue(np.all(np.abs(_F - F) < 0.00001))
            self.assertTrue(np.all(np.abs(_dF - dF) < 0.00001))

    def test_numerical_differentiation_gradient(self):
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
            self.assertTrue(np.all(np.abs(_dF - dF) < 0.001))

            _dF = NumericalDifferentiation(jacobian=CentralJacobian()).do(problem, X, F)
            self.assertTrue(np.all(np.abs(_dF - dF) < 0.001))

            _dF = NumericalDifferentiation(jacobian=ComplexStepJacobian()).do(problem, X, F)
            self.assertTrue(np.all(np.abs(_dF - dF) < 0.001))

            # _dF = ForwardNumericalDifferentiation().do(problem, Solution(X=X[0], F=F[0]))
            # _dF = ForwardNumericalDifferentiation().do(problem, Population.new(X=X, F=F))

    def test_numerical_differentiation_hessian(self):
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
            self.assertTrue(np.all(np.abs(_dF - dF) < 0.0001))
            self.assertTrue(np.all(np.abs(_ddF - ddF) < 0.001))
            print(func)


        print("Done")


    def test_numerical_differentiation_with_gradient(self):
        np.random.seed(1)

        problem = SphereWithGradientAndConstraint()

        X = np.random.random((1, problem.n_var))
        F, G, dF, ddF, dG, ddG = problem.evaluate(X, return_values_of=["F", "G", "dF", "ddF", "dG", "ddG"])

        evaluator = Evaluator()
        _dF, _ddF, _dG, _ddG = NumericalDifferentiation(eps=None).do(problem, X, F, G=G, evaluator=evaluator, hessian=True)

        print(evaluator.n_eval)
        self.assertTrue(np.all(np.abs(_dF - dF) < 0.0001))
        self.assertTrue(np.all(np.abs(_ddF - ddF) < 0.001))
        self.assertTrue(np.all(np.abs(_dG - dG) < 0.0001))
        self.assertTrue(np.all(np.abs(_ddG - ddG) < 0.001))



        print("Done")


if __name__ == '__main__':
    unittest.main()
