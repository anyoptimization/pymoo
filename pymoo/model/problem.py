import numpy as np


class Problem:

    def __init__(self, n_var=0, n_obj=0, n_constr=0, xl=0, xu=1, func=None):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = xl if type(xl) is np.ndarray else np.ones(n_var) * xl
        self.xu = xu if type(xu) is np.ndarray else np.ones(n_var) * xu
        self.func = func
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
