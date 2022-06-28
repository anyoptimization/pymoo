import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class Sphere(Problem):

    def __init__(self, n_var=10):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=-0, xu=1, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = anp.sum(anp.square(x - 0.5), axis=1)

    def _calc_pareto_front(self):
        return 0.0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0.5)
