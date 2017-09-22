import numpy as np

from model.problem import Problem


class ZDT(Problem):
    def __init__(self, n_dim=30):
        Problem.__init__(self)
        self.n_dim = n_dim
        self.n_constr = 0
        self.n_obj = 2
        self.func = self.evaluate_
        self.xl = np.zeros(self.n_dim)
        self.xu = np.ones(self.n_dim)


class ZDT1(ZDT):
    def evaluate_(self, x, f):
        f[0] = x[0]
        g = 1 + 9.0 / (self.n_dim - 1) * np.sum(x[1:])
        f[1] = g * (1 - pow((f[0] / g), 0.5))


class ZDT2(ZDT):
    def evaluate_(self, x, f):
        f[0] = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / (self.n_dim - 1)
        f[1] = g * (1 - np.math.pow((f[0] * 1.0 / g), 2))


class ZDT3(ZDT):
    def evaluate_(self, x, f):
        f[0] = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / (self.n_dim - 1)
        f[1] = g * (1 - np.math.pow(f[0] * 1.0 / g, 0.5) - (f[0] * 1.0 / g) * np.math.sin(10 * np.math.pi * f[0]))


class ZDT4(ZDT):
    def evaluate_(self, x, f):
        f[0] = x[0]
        c = 0
        for i in range(1, self.n_dim):
            c += np.math.pow(x[i], 2) - 10 * np.math.cos(4 * np.math.pi * x[i])
        g = 1 + 10 * (self.n_dim - 1) + c
        f[1] = g * (1 - np.math.sqrt(f[0] * 1.0 / g))


class ZDT6(ZDT):
    def evaluate_(self, x, f):
        f[0] = 1 - np.math.exp(-4 * x[0]) * np.math.pow(np.math.sin(6 * np.math.pi * x[0]), 6)
        g = 1 + 9.0 * np.math.pow(sum(x[1:]) / (self.n_dim - 1.0), 0.25)
        f[1] = g * (1 - np.math.pow(f[0] * 1.0 / g, 2))
