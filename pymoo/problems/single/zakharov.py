import autograd.numpy as np

from pymoo.model.problem import Problem


class Zakharov(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-10, xu=10, type_var=np.double)

    def _evaluate(self, x, out, *args, **kwargs):
        a = np.sum(0.5 * np.arange(1, self.n_var + 1) * x, axis=1)
        out["F"] = np.sum(np.square(x), axis=1) + np.square(a) + np.power(a, 4)
