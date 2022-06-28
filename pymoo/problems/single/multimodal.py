import pymoo.gradient.toolbox as anp

from pymoo.core.problem import Problem


class MultiModalSimple1(Problem):
    def __init__(self, n_var=1):
        super().__init__(n_var=n_var, n_obj=1, xl=0, xu=1, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 1 - anp.exp(-x ** 2) * anp.sin(2 * anp.pi * x) ** 2


class MultiModalSimple2(Problem):
    def __init__(self, n_var=1):
        super().__init__(n_var=n_var, n_obj=1, xl=-1, xu=0, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        x = - x
        out["F"] = 1.1 - anp.exp(-2 * x) * anp.sin(5 * anp.pi * x) ** 2
