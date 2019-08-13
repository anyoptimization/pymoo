import autograd.numpy as np

from pymoo.model.problem import Problem


class Griewank(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-600, xu=600, type_var=np.double)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = 1 + 1 / 4000 * np.sum(np.power(x, 2), axis=1) \
                  - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)
