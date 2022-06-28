import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class PressureVessel(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=1, n_ieq_constr=4, vtype=float)
        self.xl = np.array([1, 1, 10.0, 10.0])
        self.xu = np.array([99, 99, 200.0, 200.0])

    def _evaluate(self, x, out, *args, **kwargs):

        d1, d2, r, L = x[:, 0]*0.0625, x[:, 1]*0.0625, x[:, 2], x[:, 3]

        out["F"] = 0.6224*d1*r*L + 1.7781*d2*r**2 + 3.1661*d1**2*L + 19.84*d1**2*r

        g1 = (-d1 + 0.0193*r)/3
        g2 = (-d2 + 0.00954*r)/3
        g3 = (-anp.pi * r ** 2 * L - 4 * anp.pi / 3 * r ** 3 + 1296000) / 1296000
        g4 = (L - 240)/240

        out["G"] = anp.column_stack([g1, g2, g3, g4])

    def _calc_pareto_front(self):
        return 5.8853e+03

    def _calc_pareto_set(self):
        return [13, 7, 42.0984456, 176.6366]
