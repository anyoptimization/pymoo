import numpy as np

from pymoo.model.problem import Problem


class SimpleMultiModal01(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=0, xu=1, type_var=np.double)

    def _evaluate(self, X, out, *args, **kwargs):
        x, y = X[:, 0], X[:, 1]
        out["F"] = - np.exp(-9 * abs(x * y)) * np.sin(3 * np.pi * x) ** 2 * np.sin(3 * np.pi * y) ** 2
