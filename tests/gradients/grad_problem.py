import numpy as np

from pymoo.core.problem import Problem
from pymoo.problems.multi.zdt import ZDT


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
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=float)
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
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=float)

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
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=float)

            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -0.5 * np.sqrt(g / x[:, 0]) - np.sin(10 * np.pi * x[:, 0]) - 10 * np.pi * x[:, 0] * np.cos(
                10 * np.pi * x[:, 0])
            dF[:, 1, 1:] = (9 / (self.n_var - 1)) * (1 - 0.5 * np.sqrt(x[:, 0] / g))[:, None]
            out["dF"] = dF


class MySphere(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=-1, xu=+1, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum()
        out["G"] = (x ** 2).sum()


class SphereWithGradientAndConstraint(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=-1, xu=+1, elementwise_evaluation=True, autograd=False)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum()
        out["G"] = ((x - 2) ** 2).sum()

        if "dF" in out:
            out["dF"] = np.array(2 * x)[None, None, :]

        if "ddF" in out:
            out["ddF"] = np.array([[2, 0], [0, 2]])[None, None, :]

        if "dG" in out:
            out["dG"] = np.array(2 * (x - 2))[None, None, :]

        if "ddG" in out:
            out["ddG"] = np.array([[2, 0], [0, 2]])[None, None, :]


class AutomaticDifferentiationProblem(Problem):

    def __init__(self, func, n_var=2, **kwargs):
        super().__init__(n_var, n_obj=1, n_constr=0, xl=-10, xu=10, type_var=np.double, elementwise_evaluation=True,
                         evaluation_of=["F", "dF", "ddF"], **kwargs)
        self.func = func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.func(x)

        import numdifftools as nd
        out["dF"] = nd.Gradient(self.func)(x)[None, None, :]
        out["ddF"] = nd.Hessian(self.func)(x)[None, None, :]
