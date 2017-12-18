

import numpy as np

from pymoo.model.problem import Problem


class DTLZ(Problem):
    def __init__(self, n_obj=2, k=5):
        Problem.__init__(self)
        self.n_obj = n_obj
        self.n_var = self.n_obj + k - 1
        self.n_constr = 0
        self.func = self.evaluate_
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)


