import numpy as np

from pymoo.problems.ZDT.zdt import ZDT

class ZDT1(ZDT):

    def __init__(self, n_var=30):
        ZDT.__init__(self, n_var)

    def calc_pareto_front(self):
        x1 = np.arange(0, 1.01, 0.01)
        return np.array([x1, 1 - np.sqrt(x1)]).T

    def evaluate_(self, x, f):
        f[:, 0] = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:,1:], axis=1)
        f[:, 1] = g * (1 - np.power((f[:, 0] / g), 0.5))