import autograd.numpy as anp

from pymoo.model.problem import Problem


def curve(problem, n_points=200):
    X = anp.linspace(problem.xl[0], problem.xu[0], n_points)[:, None]
    F = problem.evaluate(X)
    return anp.column_stack([X, F])


class MultiModalSimple1(Problem):
    def __init__(self, n_var=1):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0, xu=1, type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 1 - anp.exp(-x ** 2) * anp.sin(2 * anp.pi * x) ** 2


class MultiModalSimple2(Problem):
    def __init__(self, n_var=1):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-1, xu=0, type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = - x
        out["F"] = 1.1 - anp.exp(-2 * x) * anp.sin(5 * anp.pi * x) ** 2
