import numpy as np

from model.problem import Problem


class Ackley(Problem):
    def __init__(self, n_dim, a=20, b=0.2, c=2 * np.pi):
        Problem.__init__(self)
        self.n_dim = n_dim
        self.a = a
        self.b = b
        self.c = c
        self.n_constr = 0
        self.n_obj = 1
        self.func = self.evaluate_
        self.xl = -5 * np.ones(self.n_dim)
        self.xu = 5 * np.ones(self.n_dim)

    def evaluate_(self, x, f):
        sum1 = 0
        sum2 = 0
        for ii in range(self.n_dim):
            xi = x(ii)
            sum1 = sum1 + xi ^ 2
            sum2 = sum2 + np.cos(self.c * xi)

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / self.n_dim))
        term2 = -np.exp(sum2 / self.d)

        f[0] = term1 + term2 + self.a + np.exp(1)
