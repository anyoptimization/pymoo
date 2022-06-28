import pymoo.gradient.toolbox as anp

from pymoo.problems.many.dtlz import DTLZ1, DTLZ3


def constraint_dc1(X, a=5, b=0.95):
    G = b - anp.cos(a * anp.pi * X[:, 0])
    return G


def constraints_dc2(gx, a=3, b=0.9):
    G = anp.column_stack([
        b - anp.cos(gx / 100 * anp.pi * a),
        b - anp.exp (-gx / 100)
    ])
    return G


def constraints_dc3(X, gx, a=5, b=0.5):
    Ggx = b - anp.cos(a * anp.pi * gx)
    Gx = b - anp.cos(a * anp.pi * X)
    return anp.column_stack([Ggx, Gx])


class DC1DTLZ1(DTLZ1):
    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 1

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_dc1(x)


class DC1DTLZ3(DTLZ3):
    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 1

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_dc1(x)


class DC2DTLZ1(DTLZ1):
    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 2

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)
        out["G"] = constraints_dc2(g)


class DC2DTLZ3(DTLZ3):
    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 2

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)
        out["G"] = constraints_dc2(g)


class DC3DTLZ1(DTLZ1):
    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, n_ieq_constr=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)
        out["G"] = constraints_dc3(X_, g)


class DC3DTLZ3(DTLZ3):
    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, n_ieq_constr=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)
        out["G"] = constraints_dc3(X_, g)
