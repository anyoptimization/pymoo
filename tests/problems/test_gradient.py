import unittest

import numpy as np

from pymoo.problems.multi.zdt import ZDT, ZDT1, ZDT2, ZDT3


class GradientTest(unittest.TestCase):

    def test_gradient(self):
        for entry in [(ZDT1(), ZDT1WithGradient()), (ZDT2(), ZDT2WithGradient()), (ZDT3(), ZDT3WithGradient())]:
            auto_diff, correct, = entry

            X = np.random.random((100, correct.n_var))

            F, dF = correct.evaluate(X, return_values_of=["F", "dF"])
            _F, _dF = auto_diff.evaluate(X, return_values_of=["F", "dF"])

            self.assertTrue(np.all(np.abs(_F - F) < 0.00001))
            self.assertTrue(np.all(np.abs(_dF - dF) < 0.00001))


class ZDT1WithGradient(ZDT):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var, evaluation_of=["F", "dF"], **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out["F"] = np.column_stack([f1, f2])

        if "dF" in out:
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=np.float)
            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -0.5 * np.sqrt(g / x[:, 0])
            dF[:, 1, 1:] = ((9 / (self.n_var - 1)) * (1 - 0.5 * np.sqrt(x[:, 0] / g)))[:, None]
            out["dF"] = dF


class ZDT2WithGradient(ZDT):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var, evaluation_of=["F", "dF"], **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.power(x, 2)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        out["F"] = np.column_stack([f1, f2])

        if "dF" in out:
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=np.float)

            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -2 * x[:, 0] / g
            dF[:, 1, 1:] = (9 / (self.n_var - 1)) * (1 + x[:, 0] ** 2 / g ** 2)[:, None]
            out["dF"] = dF


class ZDT3WithGradient(ZDT):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var, evaluation_of=["F", "dF"], **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pareto_front = np.array([]).reshape((-1, 2))
        for r in regions:
            x1 = np.linspace(r[0], r[1], int(n_pareto_points / len(regions)))
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pareto_front = np.concatenate((pareto_front, np.array([x1, x2]).T), axis=0)
        return pareto_front

    def _evaluate(self, x, out, *args, **kwargs):

        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        out["F"] = np.column_stack([f1, f2])

        if "dF" in out:
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=np.float)

            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -0.5 * np.sqrt(g / x[:, 0]) - np.sin(10 * np.pi * x[:, 0]) - 10 * np.pi * x[:, 0] * np.cos(
                10 * np.pi * x[:, 0])
            dF[:, 1, 1:] = (9 / (self.n_var - 1)) * (1 - 0.5 * np.sqrt(x[:, 0] / g))[:, None]
            out["dF"] = dF


if __name__ == '__main__':
    unittest.main()
