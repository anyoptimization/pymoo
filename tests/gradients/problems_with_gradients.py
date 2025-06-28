
import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.problems.multi.zdt import ZDT1


class ZDT1WithGradient(ZDT1):

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


class ConstrainedZDT1(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=30, n_obj=2, n_ieq_constr=3, xl=0.5, xu=1, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
        f2 = g * (1 - anp.power((f1 / g), 0.5))

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([(x ** 3).sum(axis=1), (x ** 2).sum(axis=1), x.sum(axis=1)])


class ConstrainedZDT1WithGradient(ConstrainedZDT1):

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)

        if "dF" in out:
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=float)
            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -0.5 * np.sqrt(g / x[:, 0])
            dF[:, 1, 1:] = ((9 / (self.n_var - 1)) * (1 - 0.5 * np.sqrt(x[:, 0] / g)))[:, None]
            out["dF"] = dF

        if "dG" in out:
            out["dG"] = np.swapaxes(np.array([3 * x ** 2, 2 * x, np.ones_like(x)]), 0, 1)


class ElementwiseZDT1(ElementwiseProblem):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=0, xl=0, xu=1, vtype=float, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]
        g = 1 + 9.0 / (self.n_var - 1) * x[1:].sum()
        f2 = g * (1 - (f1 / g) ** 0.5)
        out["F"] = anp.array([f1, f2])


class MySphere(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=10, n_obj=1, xl=-1, xu=+1)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum()


class MySphereWithGradient(MySphere):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum()
        out["dF"] = 2 * x


class MyConstrainedSphere(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=1, xl=-1, xu=+1)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum()
        out["G"] = (x ** 3).sum()


class MyConstrainedSphereWithGradient(MyConstrainedSphere):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum()
        out["dF"] = 2 * x

        out["G"] = (x ** 3).sum()
        out["dG"] = 3 * x ** 2


