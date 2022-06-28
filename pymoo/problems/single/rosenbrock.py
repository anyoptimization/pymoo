import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class Rosenbrock(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=-2.048, xu=2.048, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        l = []
        for i in range(x.shape[1] - 1):
            val = 100 * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (1 - x[:, i]) ** 2
            l.append(val)
        out["F"] = anp.sum(anp.column_stack(l), axis=1)

    def _calc_pareto_front(self):
        return 0.0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 1.0)
