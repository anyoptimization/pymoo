import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class SRN(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=2, xl=-20, xu=+20, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 2 + (x[:, 0] - 2) ** 2 + (x[:, 1] - 1) ** 2
        f2 = 9 * x[:, 0] - (x[:, 1] - 1) ** 2

        g1 = x[:, 0] ** 2 + x[:, 1] ** 2 - 225
        g2 = x[:, 0] - 3 * x[:, 1] + 10

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_front(self, *args, n_points=100, **kwargs):
        ps = self.pareto_set(n_points=n_points)
        return self.evaluate(ps, return_values_of=["F"])

    def _calc_pareto_set(self, *args, n_points=100, **kwargs):
        x1 = np.full(n_points, -2.5)
        x2 = np.linspace(2.5, 14.7902, n_points)
        return np.column_stack([x1, x2])
