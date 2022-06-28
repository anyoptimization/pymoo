import numpy as np
import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem


class Griewank(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, xl=-600, xu=600, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 1 + 1 / 4000 * anp.sum(anp.power(x, 2), axis=1) \
                  - anp.prod(anp.cos(x / anp.sqrt(anp.arange(1, x.shape[1] + 1))), axis=1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)
