import unittest

import autograd.numpy as anp
import numpy as np

from pymoo.model.problem import Problem


class ElementwiseEvaluationTest(unittest.TestCase):

    def test_elementwise_evaluation(self):
        X = np.random.random((100, 2))

        F = MyProblemElementwise().evaluate(X)
        _F = MyProblem().evaluate(X)

        np.testing.assert_allclose(_F, F)

    @unittest.skip("gradient for elementwise not working yet")
    def test_elementwise_evaluation_with_gradient(self):
        X = np.random.random((100, 2))

        F, dF = MyProblem().evaluate(X, return_values_of=["F", "dF"])
        _F, _dF = MyProblemElementwise().evaluate(X, return_values_of=["F", "dF"])

        np.testing.assert_allclose(_F, F)
        self.assertTrue(dF.shape == _dF.shape)
        np.testing.assert_allclose(_dF, dF)


class MyProblemElementwise(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (2 * x).sum()


class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, elementwise_evaluation=False, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = anp.sum(2 * x, axis=1)


if __name__ == '__main__':
    unittest.main()
