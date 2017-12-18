import numpy as np

from pymoo.problems.ZDT.zdt import ZDT


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