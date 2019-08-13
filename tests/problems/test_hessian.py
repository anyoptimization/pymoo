import unittest

import autograd.numpy as anp
import numpy as np

from pymoo.model.problem import Problem


class HessianTest(unittest.TestCase):

    def test_hessian(self):
        auto_diff = MyProblem()
        correct = MyProblemWithHessian()

        np.random.seed(1)

        X = np.random.random((100, correct.n_var))
        X = np.row_stack([np.array([0.5, 0.5]), X])

        F, dF, hF = correct.evaluate(X, return_values_of=["F", "dF", "hF"])
        _F, _dF, _hF = auto_diff.evaluate(X, return_values_of=["F", "dF", "hF"])

        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))
        self.assertTrue(np.all(np.abs(_dF - dF) < 0.00001))


class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 3 * x[:, 0] ** 3 + 10 * x[:, 1] ** 4 + 4 * x[:, 0] ** 2 * x[:, 1] ** 2


class MyProblemWithHessian(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, evaluation_of=["F", "dF", "hF"], **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 3 * x[:, 0] ** 3 + 10 * x[:, 1] ** 4 + 4 * x[:, 0] ** 2 * x[:, 1] ** 2

        if "dF" in out:
            f_x1 = 9 * x[:, 0] ** 2 + 8 * x[:, 0] * x[:, 1] ** 2
            f_x2 = 40 * x[:, 1] ** 3 + 8 * x[:, 0] ** 2 * x[:, 1]
            dF = np.column_stack([f_x1, f_x2])
            out["dF"] = dF[:, None, :]

        if "hF" in out:
            f_x1_x1 = 18 * x[:, 0] + 8 * x[:, 1] ** 2
            f_x1_x2 = 16 * x[:, 0] * x[:, 1]

            f_x2_x1 = 16 * x[:, 0] * x[:, 1]
            f_x2_x2 = 120 * x[:, 1] ** 2 + 8 * x[:, 0] ** 2

            out["hF"] = np.array([[f_x1_x1, f_x1_x2], [f_x2_x1, f_x2_x2]]).swapaxes(0, 2)[:, None, ...]


if __name__ == '__main__':
    unittest.main()
