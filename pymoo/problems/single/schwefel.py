import autograd.numpy as np

from pymoo.model.problem import Problem


class Schwefel(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-500, xu=500, type_var=np.double)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 418.9829 * self.n_var - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 420.9687)
