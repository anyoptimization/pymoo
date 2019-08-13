import autograd.numpy as anp

from pymoo.model.problem import Problem
from pymoo.util.normalization import normalize


class ZDT(Problem):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=1, type_var=anp.double, **kwargs)


class ZDT1(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        return anp.array([x, 1 - anp.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
        f2 = g * (1 - anp.power((f1 / g), 0.5))

        out["F"] = anp.column_stack([f1, f2])


class ZDT2(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        return anp.array([x, 1 - anp.power(x, 2)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - anp.power((f1 * 1.0 / g), 2))

        out["F"] = anp.column_stack([f1, f2])


class ZDT3(ZDT):

    def _calc_pareto_front(self, n_points=100, flatten=True):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pf = []

        for r in regions:
            x1 = anp.linspace(r[0], r[1], int(n_points / len(regions)))
            x2 = 1 - anp.sqrt(x1) - x1 * anp.sin(10 * anp.pi * x1)
            pf.append(anp.array([x1, x2]).T)

        if not flatten:
            pf = anp.concatenate([pf[None,...] for pf in pf])
        else:
            pf = anp.row_stack(pf)

        return pf

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - anp.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * anp.sin(10 * anp.pi * f1))

        out["F"] = anp.column_stack([f1, f2])


class ZDT4(ZDT):
    def __init__(self, n_var=10):
        super().__init__(n_var)
        self.xl = -5 * anp.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 5 * anp.ones(self.n_var)
        self.xu[0] = 1.0
        self.func = self._evaluate

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        return anp.array([x, 1 - anp.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * anp.cos(4.0 * anp.pi * x[:, i])
        h = 1.0 - anp.sqrt(f1 / g)
        f2 = g * h

        out["F"] = anp.column_stack([f1, f2])


class ZDT5(ZDT):

    def __init__(self, m=11, n=5, normalize=True, **kwargs):
        self.m = m
        self.n = n
        self.normalize = normalize
        super().__init__(n_var=(30 + n * (m - 1)), **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = 1 + anp.linspace(0, 1, n_pareto_points) * 30
        pf = anp.column_stack([x, (self.m-1) / x])
        if self.normalize:
            pf = normalize(pf)
        return pf

    def _evaluate(self, x, out, *args, **kwargs):

        _x = [x[:, :30]]
        for i in range(self.m - 1):
            _x.append(x[:, 30 + i * self.n: 30 + (i + 1) * self.n])

        u = anp.column_stack([x_i.sum(axis=1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[:, 1:].sum(axis=1)

        f1 = 1 + u[:, 0]
        f2 = g * (1 / f1)

        if self.normalize:
            f1 = normalize(f1, 1, 31)
            f2 = normalize(f2, (self.m-1) * 1/31, (self.m-1))

        out["F"] = anp.column_stack([f1, f2])


class ZDT6(ZDT):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0.2807753191, 1, n_pareto_points)
        return anp.array([x, 1 - anp.power(x, 2)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1 - anp.exp(-4 * x[:, 0]) * anp.power(anp.sin(6 * anp.pi * x[:, 0]), 6)
        g = 1 + 9.0 * anp.power(anp.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
        f2 = g * (1 - anp.power(f1 / g, 2))

        out["F"] = anp.column_stack([f1, f2])
