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
    def __init__(self, n_var=30):
        ZDT.__init__(self, n_var)

    def calc_pareto_front(self):
        x1 = np.arange(0, 1.01, 0.01)
        return np.array([x1, 1 - np.sqrt(x1)]).T

    def evaluate_(self, x, f):
        f[:, 0] = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:,1:], axis=1)
        f[:, 1] = g * (1 - np.power((f[:, 0] / g), 0.5))


class ZDT2(ZDT):
    def calc_pareto_front(self):
        x1 = np.arange(0, 1.01, 0.01)
        return np.array([x1, 1 - np.power(x1, 2)]).T

    def evaluate_(self, x, f):
        f[:, 0] = x[:, 0]
        c = np.sum(x[:,1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f[:, 1] = g * (1 - np.power((f[:, 0] * 1.0 / g), 2))


class ZDT3(ZDT):
    def calc_pareto_front(self):
        regions = [[0, 0.0830015349], [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041], [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pareto_front = np.array([]).reshape((-1, 2))
        for r in regions:
            x1 = np.linspace(r[0], r[1], 50)
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pareto_front = np.concatenate((pareto_front, np.array([x1, x2]).T), axis=0)
        return pareto_front

    def evaluate_(self, x, f):
        f[:, 0] = x[:, 0]
        c = np.sum(x[:,1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f[:,1] = g * (1 - np.power(f[:,0] * 1.0 / g, 0.5) - (f[:,0] * 1.0 / g) * np.sin(10 * np.pi * f[:,0]))


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
