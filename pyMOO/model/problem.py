import numpy as np


class Problem:

    def __init__(self, n_dim=0, n_obj=0, n_constr=0, xl=0, xu=1, func=None):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = xl if type(xl) is np.ndarray else np.ones(n_dim) * xl
        self.xu = xu if type(xu) is np.ndarray else np.ones(n_dim) * xu
        self.func = func

    def evaluate(self, x):

        f = np.zeros(self.n_obj)
        g = np.zeros(self.n_constr)

        if self.n_constr > 0:
            self.func(x, f, g)
        else:
            self.func(x, f)

        return f, g

    def name(self):
        return self.__class__.__name__
