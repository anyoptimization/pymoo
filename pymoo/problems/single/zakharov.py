import autograd.numpy as anp

from pymoo.model.problem import Problem


class Zakharov(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-10, xu=10, type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        a = anp.sum(0.5 * anp.arange(1, self.n_var + 1) * x, axis=1)
        out["F"] = anp.sum(anp.square(x), axis=1) + anp.square(a) + anp.power(a, 4)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return anp.full(self.n_var, 0)
