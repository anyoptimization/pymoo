import autograd.numpy as anp

from pymoo.model.problem import Problem


class Rastrigin(Problem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-5, xu=5, type_var=anp.double)
        self.A = A

    def _evaluate(self, x, out, *args, **kwargs):
        z = anp.power(x, 2) - self.A * anp.cos(2 * anp.pi * x)
        out["F"] = self.A * self.n_var + anp.sum(z, axis=1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return anp.full(self.n_var, 0)
