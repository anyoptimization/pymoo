import numpy as np

from model.problem import Problem


class ZDT(Problem):
    def __init__(self, n_var=30):
        Problem.__init__(self, func=self.evaluate_)
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = 2

        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)


class ZDT1(ZDT):
    def evaluate_(self, x, f):
        f[0] = x[0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[1:])
        f[1] = g * (1 - pow((f[0] / g), 0.5))


class ZDT2(ZDT):
    def evaluate_(self, x, f):
        f[0] = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f[1] = g * (1 - np.math.pow((f[0] * 1.0 / g), 2))


class ZDT3(ZDT):
    def evaluate_(self, x, f):
        f[0] = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f[1] = g * (1 - np.math.pow(f[0] * 1.0 / g, 0.5) - (f[0] * 1.0 / g) * np.math.sin(10 * np.math.pi * f[0]))


class ZDT4(ZDT):
    def __init__(self, n_var=10):
        ZDT.__init__(self, n_var)
        self.xl = -5 * np.ones(self.n_var)
        self.xu = 5 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        f[0] = x[0]
        g = 1 + 10 * (self.n_var - 1) + np.sum(np.math.pow(x, 2) - np.math.cos(4 * np.math.pi * x))
        f[1] = g * (1 - np.math.sqrt(f[0] * 1.0 / g))


class ZDT6(ZDT):
    def __init__(self, n_var=10):
        ZDT.__init__(self, n_var)
        self.func = self.evaluate_

    def evaluate_(self, x, f):
        f[0] = 1 - np.math.exp(-4 * x[0]) * np.math.pow(np.math.sin(6 * np.math.pi * x[0]), 6)
        g = 1 + 9.0 * np.math.pow(sum(x[1:]) / (self.n_var - 1.0), 0.25)
        f[1] = g * (1 - np.math.pow(f[0] / g, 2))
