import autograd.numpy as anp

from pymoo.problems.many.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4


class C1DTLZ1(DTLZ1):

    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c1_linear(out["F"])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        return super()._calc_pareto_front(ref_dirs, *args, **kwargs)


class C1DTLZ3(DTLZ3):

    def __init__(self, n_var=12, n_obj=3, r=None, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

        if r is None:
            if self.n_obj < 5:
                r = 9.0
            elif 5 <= self.n_obj <= 12:
                r = 12.5
            else:
                r = 15.0

        self.r = r

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c1_spherical(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        return super()._calc_pareto_front(ref_dirs, *args, **kwargs)


class C2DTLZ2(DTLZ2):

    def __init__(self, n_var=12, n_obj=3, r=None, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

        if r is None:
            if n_obj == 2:
                r = 0.2
            elif n_obj == 3:
                r = 0.4
            else:
                r = 0.5

        self.r = r

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c2(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        F = super()._calc_pareto_front(ref_dirs, *args, **kwargs)
        G = constraint_c2(F, r=self.r)
        G[G <= 0] = 0
        return F[G <= 0]


class C3DTLZ1(DTLZ1):

    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = n_obj

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c3_linear(out["F"])


class C3DTLZ4(DTLZ4):

    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = n_obj

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c3_spherical(out["F"])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        F = super()._calc_pareto_front(ref_dirs, *args, **kwargs)
        a = anp.sqrt(anp.sum(F ** 2, 1) - 3 / 4 * anp.max(F ** 2, axis=1))
        a = anp.expand_dims(a, axis=1)
        a = anp.tile(a, [1, ref_dirs.shape[1]])
        F = F / a

        return F


def constraint_c1_linear(f):
    g = - (1 - f[:, -1] / 0.6 - anp.sum(f[:, :-1] / 0.5, axis=1))

    return g


def constraint_c1_spherical(f, r):
    radius = anp.sum(f ** 2, axis=1)
    g = - (radius - 16) * (radius - r ** 2)

    return g


def constraint_c2(f, r):
    n_obj = f.shape[1]

    v1 = anp.inf * anp.ones(f.shape[0])

    for i in range(n_obj):
        temp = (f[:, i] - 1) ** 2 + (anp.sum(f ** 2, axis=1) - f[:, i] ** 2) - r ** 2
        v1 = anp.minimum(temp.flatten(), v1)

    a = 1 / anp.sqrt(n_obj)
    v2 = anp.sum((f - a) ** 2, axis=1) - r ** 2
    g = anp.minimum(v1, v2.flatten())

    return g


def constraint_c3_linear(f):  # M lines
    n_obj = f.shape[1]
    g = []

    for i in range(n_obj):
        _g = 1 - f[:, i] / 0.5 - (anp.sum(f, axis=1) - f[:, i])
        g.append(_g)

    return anp.column_stack(g)


def constraint_c3_spherical(f):  # M ellipse
    n_obj = f.shape[1]
    g = []

    for i in range(n_obj):
        _g = 1 - f[:, i] ** 2 / 4 - (anp.sum(f ** 2, axis=1) - f[:, i] ** 2)
        g.append(_g)
    return anp.column_stack(g)


def constraint_c4_cylindrical(f, r):  # cylindrical
    l = anp.mean(f, axis=1)
    l = anp.expand_dims(l, axis=1)
    g = -anp.sum(anp.power(f - l, 2), axis=1) + anp.power(r, 2)
    return g