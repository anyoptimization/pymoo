import numpy as np

from pymoo.problems.ZDT.zdt import ZDT


class ZDT4(ZDT):
    def __init__(self, n_var=10):
        ZDT.__init__(self, n_var)
        self.xl = -5 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 5 * np.ones(self.n_var)
        self.xu[0] = 1.0
        self.func = self.evaluate_

    def calc_pareto_front(self):
        x1 = np.arange(0, 1.01, 0.01)
        return np.array([x1, 1 - np.sqrt(x1)]).T

    def evaluate_(self, x, f):
        f[:, 0] = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * np.cos(4.0 * np.pi * x[:, i])
        h = 1.0 - np.sqrt(f[:, 0] / g)
        f[:, 1] = g * h