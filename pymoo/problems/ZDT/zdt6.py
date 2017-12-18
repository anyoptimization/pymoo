import numpy as np

from pymoo.problems.ZDT.zdt import ZDT


class ZDT6(ZDT):
    def __init__(self, n_var=10):
        ZDT.__init__(self, n_var)
        self.func = self.evaluate_

    def calc_pareto_front(self):
        x1 = np.linspace(0.2807753191, 1, 100)
        return np.array([x1, 1 - np.power(x1, 2)]).T

    def evaluate_(self, x, f):
        f[:, 0] = 1 - np.exp(-4 * x[:, 0]) * np.power(np.sin(6 * np.pi * x[:, 0]), 6)
        g = 1 + 9.0 * np.power(np.sum(x[:,1:], axis=1) / (self.n_var - 1.0), 0.25)
        f[:, 1] = g * (1 - np.power(f[:, 0] / g, 2))