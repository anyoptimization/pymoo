import numpy as np

from pymoo.model.problem import Problem


class ZDT(Problem):
    def __init__(self, n_var=30):
        Problem.__init__(self, func=self.evaluate_)
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 2

        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)


