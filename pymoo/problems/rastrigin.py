import numpy as np

from model.problem import Problem


class Rastrigin(Problem):
    def __init__(self,n_var=2, A=10):
        Problem.__init__(self)
        self.n_var = n_var
        self.A = A
        self.n_constr = 0
        self.n_obj = 1
        self.func = self.evaluate_
        self.xl = -5 * np.ones(self.n_var)
        self.xu = 5 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        f[:,0] = self.A * self.n_var + np.sum(z, axis=1)
