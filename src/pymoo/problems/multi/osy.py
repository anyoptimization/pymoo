import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote


class OSY(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=6, vtype=float)
        self.xl = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.xu = np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0])

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = - (25 * (x[:, 0] - 2) ** 2 + (x[:, 1] - 2) ** 2 + (x[:, 2] - 1) ** 2 + (x[:, 3] - 4) ** 2 + (
                x[:, 4] - 1) ** 2)
        f2 = anp.sum(anp.square(x), axis=1)

        g1 = (x[:, 0] + x[:, 1] - 2.0) / 2.0
        g2 = (6.0 - x[:, 0] - x[:, 1]) / 6.0
        g3 = (2.0 - x[:, 1] + x[:, 0]) / 2.0
        g4 = (2.0 - x[:, 0] + 3.0 * x[:, 1]) / 2.0
        g5 = (4.0 - (x[:, 2] - 3.0) ** 2 - x[:, 3]) / 4.0
        g6 = ((x[:, 4] - 3.0) ** 2 + x[:, 5] - 4.0) / 4.0

        out["F"] = anp.column_stack([f1, f2])

        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6])
        out["G"] = - out["G"]

    def _calc_pareto_front(self):
        return Remote.get_instance().load("pymoo", "pf", "osy.pf")
