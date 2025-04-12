import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class WRM(Problem):

    def __init__(self):
        xl = np.array([0.01, 0.01, 0.01])
        xu = np.array([0.45, 0.10, 0.10])
        super().__init__(n_var=3, n_obj=5, n_ieq_constr=7, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):

        # the definition is index 1 based -> simply add a dummy var in the beginning
        x = anp.column_stack([anp.zeros((len(x), 1)), x])

        f1 = 106780.37 * (x[:, 2] + x[:, 3]) + 61704.67
        f2 = 3000 * x[:, 1]
        f3 = 305700 * 2289 * x[:, 2] / (0.06 * 2289) ** 0.65
        f4 = 250 * 2289 * anp.exp(-39.75 * x[:, 2] + 9.9 * x[:, 3] + 2.74)
        f5 = 25 * (1.39 / (x[:, 1] * x[:, 2]) + 4940 * x[:, 3] - 80)

        deno = 1. / (x[:, 1] * x[:, 2])

        g1 = -(1.0 - (0.00139 * deno + 4.94 * x[:, 3] - 0.08))
        g2 = -(1.0 - (0.000306 * deno + 1.082 * x[:, 3] - 0.0986))
        g3 = -(50000 - (12.307 * deno + 49408.24 * x[:, 3] + 4051.02))
        g4 = -(16000 - (2.098 * deno + 8046.33 * x[:, 3] - 696.71))
        g5 = -(10000 - (2.138 * deno + 7883.39 * x[:, 3] - 705.04))
        g6 = -(2000 - (0.417 * deno + 1721.26 * x[:, 3] - 136.54))
        g7 = -(550 - (0.164 * deno + 631.13 * x[:, 3] - 54.58))

        out["F"] = anp.column_stack([f1, f2, f3, f4, f5])
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7])
