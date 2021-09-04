import unittest

import autograd.numpy as anp
import numpy as np

from pymoo.core.problem import Problem, ElementwiseProblem


class MyProblemElementwise(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (2 * x).sum()


class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = anp.sum(2 * x, axis=1)


def test_elementwise_evaluation():
    X = np.random.random((100, 2))

    F = MyProblemElementwise().evaluate(X)
    _F = MyProblem().evaluate(X)

    np.testing.assert_allclose(_F, F)


if __name__ == '__main__':
    unittest.main()
