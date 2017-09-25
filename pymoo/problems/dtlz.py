import functools

import numpy as np

from model.problem import Problem


class DTLZ(Problem):
    def __init__(self, n_obj=2, k=5):
        Problem.__init__(self)
        self.n_obj = n_obj
        self.n_var = self.n_obj + k - 1
        self.n_constr = 0
        self.func = self.evaluate_
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)



class DTLZ1(DTLZ):

    def evaluate_(self, x, f):

        g = 0
        for j in range(self.n_obj - 1, self.n_var):
            g += 100.0 * (1.0 + np.power((x[j] - 0.5), 2) - np.math.cos(20.0 * np.math.pi * (x[j] - 0.5)))

        f[0] = 0.5 * np.prod(x[:(self.n_obj - 1)]) * (1.0 + g)
        for j in range(1, self.n_obj - 1):
            f[j] = 0.5 * np.prod(x[:(self.n_obj - j - 1)]) * (1 - x[self.n_obj - j]) * (1 + g)

        f[self.n_obj - 1] = 0.5 * (1 - x[0]) * (1 + g)
