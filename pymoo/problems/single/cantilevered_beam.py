import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class CantileveredBeam(Problem):

    def __init__(self):
        super().__init__(n_var=4, n_obj=1, n_ieq_constr=2, vtype=float)
        self.xl = np.array([2, 0.1, 0.1, 3.0])
        self.xu = np.array([12.0, 1.0, 2.0, 7.0])
        self.h1 = np.array([0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0])

    def _evaluate(self, x, out, *args, **kwargs):
        E, L, P = 1e7, 36.0, 1000.0

        b1, h1, b2, H = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        I = 1 / 12 * b2 * (H - 2 * h1) ** 3 + 2 * (1 / 12 * b1 * h1 ** 3 + b1 * h1 * (H - h1) ** 2 / 4)
        volume = (2 * h1 * b1 + (H - 2 * h1) * b2) * L
        out["F"] = volume

        sigma = P * L * H / (2 * I)
        delta = P * L ** 3 / (3 * E * I)

        g1 = (sigma - 5000.0) / 5000.0
        g2 = (delta - 0.1) / 0.1
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_front(self):
        return 92.7693

    def _calc_pareto_set(self):
        return [9.4846, 0.1000, 0.1000, 7.0000]
