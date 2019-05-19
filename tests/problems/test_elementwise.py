import unittest

import numpy as np

from pymoo.model.problem import Problem


class ElementwiseEvaluationTest(unittest.TestCase):

    def test_elementwise_evaluation(self):
        X = np.random.random((100, 2))

        F = MyProblemElementwise().evaluate(X)
        _F = MyProblem().evaluate(X)

        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))

    def test_elementwise_evaluation_with_gradient(self):
        X = np.random.random((100, 2))

        F, dF = MyProblem().evaluate(X, return_values_of=["F", "dF"])
        _F, _dF = MyProblemElementwise().evaluate(X, return_values_of=["F", "dF"])

        self.assertTrue(dF.shape == _dF.shape)
        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))
        self.assertTrue(np.all(np.abs(_dF - dF) < 0.00001))


class MyProblemElementwise(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x.sum()


class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, elementwise_evaluation=False, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x.sum(axis=1)


if __name__ == '__main__':
    unittest.main()
