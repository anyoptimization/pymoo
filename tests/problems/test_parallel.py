import unittest

import numpy as np

from pymoo.model.problem import Problem


class ParallelEvaluationTest(unittest.TestCase):

    def get_data(self):
        np.random.seed(1)
        X = np.random.random((100, 2))
        F = X.sum(axis=1)[:, None]
        return X, F

    def test_evaluation_in_threads(self):
        X, F = self.get_data()
        _F = MyProblemElementwise(parallelization="threads").evaluate(X)
        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))

    def test_evaluation_in_threads_number(self):
        X, F = self.get_data()
        _F = MyProblemElementwise(parallelization=("threads", 2)).evaluate(X)
        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))


class MyProblemElementwise(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x.sum()


if __name__ == '__main__':
    unittest.main()
