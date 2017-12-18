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

    def ideal_point(self):
        return np.min(self.pareto_front(), axis=0)

    def pareto_front(self):
        if self._pareto_front is None:
            self._pareto_front = self.calc_pareto_front()
        return self._pareto_front

    def evaluate(self, x):

        only_single_value = len(np.shape(x)) == 1
        if only_single_value:
            x = np.array([x])

        if x.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' % (x.shape[1], self.n_var))

        f = np.zeros((x.shape[0], self.n_obj))
        g = np.zeros((x.shape[0], self.n_constr))

        if self.n_constr > 0:
            self.func(x, f, g)
        else:
            self.func(x, f)

        if only_single_value:
            return f[0, :], g[0, :]

        return f, g

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_constr: %s\n" % self.n_constr
        s += "# f(xl): %s\n" % self.evaluate(self.xl)[0]
        s += "# f((xl+xu)/2): %s\n" % self.evaluate((self.xl + self.xu) / 2.0)[0]
        s += "# f(xu): %s\n" % self.evaluate(self.xu)[0]
        return s

