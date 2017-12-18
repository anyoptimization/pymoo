import numpy as np

from pymoo.problems.ZDT.zdt import ZDT


class ZDT2(ZDT):
    def calc_pareto_front(self):
        x1 = np.arange(0, 1.01, 0.01)
        return np.array([x1, 1 - np.power(x1, 2)]).T

    def evaluate_(self, x, f):
        f[:, 0] = x[:, 0]
        c = np.sum(x[:,1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f[:, 1] = g * (1 - np.power((f[:, 0] * 1.0 / g), 2))