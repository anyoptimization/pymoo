import numpy as np


class Problem:
    def __init__(self, n_var=0, n_obj=0, n_constr=0, xl=0, xu=1, func=None):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = xl if type(xl) is np.ndarray else np.ones(n_var) * xl
        self.xu = xu if type(xu) is np.ndarray else np.ones(n_var) * xu
        self.func = func
        self._pareto_front = None

    def nadir_point(self):
        return np.max(self.pareto_front(), axis=0)

    def pareto_front(self):
        if self._pareto_front is None:
            self._pareto_front = self.calc_pareto_front()
        return self._pareto_front

    def evaluate_single(self, x):
        f = np.zeros(self.n_obj)
        g = np.zeros(self.n_constr)

        if self.n_constr > 0:
            self.func(x, f, g)
        else:
            self.func(x, f)

        return f, g

    def evaluate(self, x, as_matrix=False):

        if as_matrix:
            n = np.shape(x)[0]
            f = np.zeros((n, self.n_obj))
            g = np.zeros((n, self.n_constr))
            for i in range(n):
                f[i, :], g[i, :] = self.evaluate_single(x[i])
            return f,g
        else:
            return self.evaluate_single(x)

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_constr: %s\n" % self.n_constr
        s += "# f(xl): %s\n" % self.evaluate(self.xl)[0]
        s += "# f((xl+xu)/2): %s\n" % self.evaluate((self.xl+self.xu)/2.0)[0]
        s += "# f(xu): %s\n" % self.evaluate(self.xu)[0]
        return s
