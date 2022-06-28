import pymoo.gradient.toolbox as anp

from pymoo.core.problem import Problem


class SimpleMultiModal01(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=0, xl=0, xu=1, vtype=float)

    def _evaluate(self, X, out, *args, **kwargs):
        x, y = X[:, 0], X[:, 1]
        out["F"] = - anp.exp(-9 * abs(x * y)) * anp.sin(3 * anp.pi * x) ** 2 * anp.sin(3 * anp.pi * y) ** 2
