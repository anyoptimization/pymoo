import numpy as np


class Problem:

    def __init__(self):
        self.n_dim = 0
        self.n_obj = 0
        self.n_constr = 0
        self.xl = []
        self.xu = []
        self.func = None

    def evaluate(self, x):
        f = np.zeros(self.n_obj)
        g = np.zeros(self.n_constr)

        if self.n_constr > 0:
            self.func(x, f, g)
        else:
            self.func(x, f)
        return f, g
