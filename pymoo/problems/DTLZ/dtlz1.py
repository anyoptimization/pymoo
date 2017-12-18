import numpy as np

from pymoo.problems.DTLZ.dtlz import DTLZ


class DTLZ1(DTLZ):

    def calc_pareto_front(self):
        x1 = np.arange(0, 0.5, 100)
        return np.array([x1, 0.5 - x1]).T

    def evaluate_(self, x, f):

        g = 0
        for j in range(self.n_obj - 1, self.n_var):
            g += 100.0 * (1.0 + np.power((x[j] - 0.5), 2) - np.math.cos(20.0 * np.math.pi * (x[j] - 0.5)))

        f[0] = 0.5 * np.prod(x[:(self.n_obj - 1)]) * (1.0 + g)
        for j in range(1, self.n_obj - 1):
            f[j] = 0.5 * np.prod(x[:(self.n_obj - j - 1)]) * (1 - x[self.n_obj - j]) * (1 + g)

        f[self.n_obj - 1] = 0.5 * (1 - x[0]) * (1 + g)