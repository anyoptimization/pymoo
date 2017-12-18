import numpy as np
from pymoo.model.problem import Problem


class Kursawe(Problem):
    def __init__(self):
        Problem.__init__(self)
        self.n_var = 3
        self.n_constr = 0
        self.n_obj = 2
        self.func = self.evaluate_
        self.xl = -5 * np.ones(self.n_var)
        self.xu = 5 * np.ones(self.n_var)

    def evaluate_(self, x, f):
        f[0] = np.sum([-10 * np.exp(-0.2 * np.sqrt(x[i] * x[i] + x[i + 1] * x[i + 1])) for i in range(2)])
        f[1] = np.sum([pow(abs(x[i]), 0.8) + 5 * np.sin(pow(x[i], 3)) for i in range(0, 3)])
