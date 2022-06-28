import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote


class WeldedBeam(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=4, vtype=float)
        self.xl = np.array([0.125, 0.1, 0.1, 0.125])
        self.xu = np.array([5.0, 10.0, 10.0, 5.0])

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1.10471 * x[:, 0] ** 2 * x[:, 1] + 0.04811 * x[:, 2] * x[:, 3] * (14.0 + x[:, 1])
        f2 = 2.1952 / (x[:, 3] * x[:, 2] ** 3)

        P = 6000
        L = 14
        t_max = 13600
        s_max = 30000

        R = anp.sqrt(0.25 * (x[:, 1] ** 2 + (x[:, 0] + x[:, 2]) ** 2))
        M = P * (L + x[:, 1] / 2)
        J = 2 * anp.sqrt(0.5) * x[:, 0] * x[:, 1] * (x[:, 1] ** 2 / 12 + 0.25 * (x[:, 0] + x[:, 2]) ** 2)
        t1 = P / (anp.sqrt(2) * x[:, 0] * x[:, 1])
        t2 = M * R / J
        t = anp.sqrt(t1 ** 2 + t2 ** 2 + t1 * t2 * x[:, 1] / R)
        s = 6 * P * L / (x[:, 3] * x[:, 2] ** 2)
        P_c = 64746.022 * (1 - 0.0282346 * x[:, 2]) * x[:, 2] * x[:, 3] ** 3

        g1 = (1 / t_max) * (t - t_max)
        g2 = (1 / s_max) * (s - s_max)
        g3 = (1 / (5 - 0.125)) * (x[:, 0] - x[:, 3])
        g4 = (1 / P) * (P - P_c)

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2, g3, g4])

    def _calc_pareto_front(self):
        return Remote.get_instance().load("pymoo", "pf", "welded_beam.pf")
