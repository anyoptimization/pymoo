import numpy as np

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.problems.dyn import DynamicTestProblem
from pymoo.util.remote import Remote


class DF(DynamicTestProblem):

    def __init__(self, n_var=10, nt=10, taut=20, **kwargs):
        super().__init__(nt,
                         taut,
                         n_var=n_var,
                         n_obj=2,
                         xl=0,
                         xu=1,
                         **kwargs)

    def _calc_pareto_front(self, *args, **kwargs):
        return Remote.get_instance().load("pymoo", "pf", "DF", f"{str(self.__class__.__name__)}.pf")


class DF1(DF):

    def _evaluate(self, x, out, *args, **kwargs):
        v = np.sin(0.5 * np.pi * self.time)

        G = np.abs(v)
        H = 0.75 * v + 1.25
        g = 1 + np.sum((x[:, 1:] - G) ** 2, axis=1)

        f1 = x[:, 0]
        f2 = g * (1 - ((f1 / g) ** H))

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        v = np.sin(0.5 * np.pi * self.time)
        H = 0.75 * v + 1.25

        f1 = np.linspace(0, 1, n_pareto_points)
        return np.array([f1, 1 - f1 ** H]).T


class DF2(DF):
    def _evaluate(self, x, out, *args, **kwargs):
        v = np.sin(0.5 * np.pi * self.time)
        G = np.abs(v)

        n = self.n_var
        r = int((n - 1) * G)
        not_r = [k for k in range(n) if k != r]

        g = 1 + np.sum((x[:, not_r] - G) ** 2, axis=1)

        f1 = x[:, r]
        f2 = g * (1 - np.power(f1 / g, 0.5))

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        f1 = np.linspace(0, 1, n_pareto_points)
        return np.array([f1, 1 - np.sqrt(f1)]).T


class DF3(DF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[1:] = -1.0
        self.xu[1:] = 2.0

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.sin(0.5 * np.pi * self.time)
        H = G + 1.5

        g = 1 + np.sum((x[:, 1:] - G - x[:, [0]] ** H) ** 2, axis=1)
        f1 = x[:, 0]
        f2 = g * (1 - (x[:, 0] / g) ** H)

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        g = 1

        G = np.sin(np.dot(np.dot(0.5, np.pi), self.time))
        H = G + 1.5
        f1 = np.copy(x)
        f2 = np.dot(g, (1 - (x / g) ** H))

        return np.column_stack([f1, f2])


class DF4(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[:] = -2.0
        self.xu[:] = +2.0

    def _evaluate(self, x, out, *args, **kwargs):
        n = self.n_var
        a = np.sin(0.5 * np.pi * self.time)
        b = 1 + np.abs(np.cos(0.5 * np.pi * self.time))
        H = 1.5 + a
        c = np.maximum(np.abs(a), a + b)

        g = 1.0
        for i in range(1, n):
            g += (x[:, i] - (a * (x[:, 0] / c) ** 2 / (i + 1))) ** 2

        f1 = g * np.abs(x[:, 0] - a) ** H
        f2 = g * np.abs(x[:, 0] - a - b) ** H

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        a = np.sin(0.5 * np.pi * self.time)
        b = 1 + np.abs(np.cos(0.5 * np.pi * self.time))
        H = 1.5 + a
        x = np.linspace(a, a + b, n_pareto_points)

        f1 = np.abs(x - a) ** H
        f2 = np.abs(x - a - b) ** H

        return np.array([f1, f2]).T


class DF5(DF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[1:] = -1.0
        self.xu[1:] = +1.0

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.sin(0.5 * np.pi * self.time)
        w = np.floor(10 * G)
        g = 1 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f1 = g * (x[:, 0] + 0.02 * np.sin(w * np.pi * x[:, 0]))
        f2 = g * (1 - x[:, 0] + 0.02 * np.sin(w * np.pi * x[:, 0]))

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        G = np.sin(0.5 * np.pi * self.time)
        w = np.floor(10 * G)
        f1 = x + 0.02 * np.sin(w * np.pi * x)
        f2 = 1 - x + 0.02 * np.sin(w * np.pi * x)
        return np.array([f1, f2]).T


class DF6(DF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[1:] = -1.0
        self.xu[1:] = +1.0

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.sin(0.5 * np.pi * self.time)
        a = 0.2 + 2.8 * np.abs(G)
        y = x[:, 1:] - G
        g = 1 + np.sum((np.abs(G) * y ** 2 - 10 * np.cos(2 * np.pi * y) + 10), axis=1)

        f1 = g * np.power(x[:, 0] + 0.1 * np.sin(3 * np.pi * x[:, 0]), a)
        f2 = g * np.power(1 - x[:, 0] + 0.1 * np.sin(3 * np.pi * x[:, 0]), a)

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        G = np.sin(0.5 * np.pi * self.time)
        a = 0.2 + 2.8 * np.abs(G)
        f1 = (x + 0.1 * np.sin(3 * np.pi * x)) ** a
        f2 = (1 - x + 0.1 * np.sin(3 * np.pi * x)) ** a

        return np.array([f1, f2]).T


class DF7(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[0] = 1.0
        self.xu[0] = 4.0

    def _evaluate(self, x, out, *args, **kwargs):
        a = 5 * np.cos(0.5 * np.pi * self.time)
        g = 1 + np.sum((x[:, 1:] - 1 / (1 + np.exp(a * (x[:, [0]] - 2.5)))) ** 2, axis=1)

        f1 = g * (1 + self.time) / x[:, 0]
        f2 = g * x[:, 0] / (1 + self.time)

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(1, 4, n_pareto_points)
        f1 = (1 + self.time) / x
        f2 = x / (1 + self.time)
        pf = np.array([f1, f2]).T
        pf = pf[np.argsort(pf[:, 0])]
        return pf


class DF8(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[1:] = -1.0
        self.xu[1:] = +1.0

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.sin(0.5 * np.pi * self.time)
        a = 2.25 + 2 * np.cos(2 * np.pi * self.time)
        b = 1
        tmp = G * np.sin(4 * np.pi * np.power(x[:, 0].reshape(len(x), 1), b)) / (1 + np.abs(G))
        g = 1 + np.sum((x[:, 1:] - tmp) ** 2, axis=1)
        f1 = g * (x[:, 0] + 0.1 * np.sin(3 * np.pi * x[:, 0]))
        f2 = g * np.power(1 - x[:, 0] + 0.1 * np.sin(3 * np.pi * x[:, 0]), a)

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        a = 2.25 + 2 * np.cos(2 * np.pi * self.time)

        f1 = x + 0.1 * np.sin(3 * np.pi * x)
        f2 = (1 - x + 0.1 * np.sin(3 * np.pi * x)) ** a

        return np.array([f1, f2]).T


class DF9(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[1:] = -1.0
        self.xu[1:] = +1.0

    def _evaluate(self, x, out, *args, **kwargs):
        _, n = x.shape
        N = 1 + np.floor(10 * np.abs(np.sin(0.5 * np.pi * self.time)))
        g = 1
        for i in range(1, n):
            tmp = x[:, i] - np.cos(4 * self.time + x[:, 0] + x[:, i - 1])
            g = g + tmp ** 2
        f1 = g * (x[:, 0] + np.maximum(0, (0.1 + 0.5 / N) * np.sin(2 * N * np.pi * x[:, 0])))
        f2 = g * (1 - x[:, 0] + np.maximum(0, (0.1 + 0.5 / N) * np.sin(2 * N * np.pi * x[:, 0])))

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        N = 1 + np.floor(10 * np.abs(np.sin(0.5 * np.pi * self.time)))

        f1 = x + np.maximum(0, (0.1 + 0.5 / N) * np.sin(2 * N * np.pi * x))
        f2 = 1 - x + np.maximum(0, (0.1 + 0.5 / N) * np.sin(2 * N * np.pi * x))

        h = get_PF(np.array([f1, f2]), True)
        return h


class DF10(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xl[2:] = -1.0
        self.xu[2:] = +1.0
        self.n_obj = 3

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.sin(0.5 * np.pi * self.time)
        H = 2.25 + 2 * np.cos(0.5 * np.pi * self.time)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)
        tmp = np.sin(4 * np.pi * (x0 + x1)) / (1 + np.abs(G))  # in the document is 2*
        g = 1 + np.sum((x[:, 2:] - tmp) ** 2, axis=1)
        g = g.reshape(len(g), 1)
        f1 = (g * np.power(np.sin(0.5 * np.pi * x0), H)).reshape(len(g), )
        f2 = (g * np.power(np.sin(0.5 * np.pi * x1) * np.cos(0.5 * np.pi * x0), H)).reshape(len(g), )
        f3 = (g * np.power(np.cos(0.5 * np.pi * x1) * np.cos(0.5 * np.pi * x0), H)).reshape(len(g), )

        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        H = 2.25 + 2 * np.cos(0.5 * np.pi * self.time)
        g = 1
        f1 = g * np.sin(0.5 * np.pi * x1) ** H
        f2 = np.multiply(g * np.sin(0.5 * np.pi * x2) ** H, np.cos(0.5 * np.pi * x1) ** H)
        f3 = np.multiply(g * np.cos(0.5 * np.pi * x2) ** H, np.cos(0.5 * np.pi * x1) ** H)

        return get_PF(np.array([f1, f2, f3]), False)


class DF11(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_obj = 3

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.abs(np.sin(0.5 * np.pi * self.time))
        g = 1 + G + np.sum((x[:, 2:] - 0.5 * G * x[:, 0].reshape(len(x), 1)) ** 2, axis=1)
        y = [np.pi * G / 6.0 + (np.pi / 2 - np.pi * G / 3.0) * x[:, i] for i in [0, 1]]

        f1 = g * np.sin(y[0])
        f2 = g * np.sin(y[1]) * np.cos(y[0])
        f3 = g * np.cos(y[1]) * np.cos(y[0])

        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = np.abs(np.sin(0.5 * np.pi * self.time))
        y1 = np.pi * G / 6 + (np.pi / 2 - np.pi * G / 3) * x1
        y2 = np.pi * G / 6 + (np.pi / 2 - np.pi * G / 3) * x2

        f1 = np.sin(y1)
        f2 = np.dot(np.multiply(1, np.sin(y2)), np.cos(y1))
        f3 = np.dot(np.multiply(1, np.cos(y2)), np.cos(y1))

        return get_PF(np.array([f1, f2, f3]), False)


class DF12(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_obj = 3
        self.xl[2:] = -1.0
        self.xu[2:] = +1.0

    def _evaluate(self, x, out, *args, **kwargs):
        k = 10 * np.sin(np.pi * self.time)
        #        r = 1 - np.modulo(k,2)
        r = 1
        x0 = x[:, 0].reshape(len(x), 1)
        tmp1 = x[:, 2:] - np.sin(self.time * x0)
        tmp2 = np.abs(np.sin(np.floor(k * (2 * x[:, 0:2] - r)) * np.pi / 2))
        g = 1 + np.sum(tmp1 ** 2, axis=1) + np.prod(tmp2)

        f1 = g * np.cos(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f2 = g * np.sin(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f3 = g * np.sin(0.5 * np.pi * x[:, 0])

        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        k = 10 * np.sin(np.pi * self.time)
        tmp2 = np.abs(
            (np.sin((np.floor(k * (2 * x1 - 1)) * np.pi) / 2) *
             np.sin((np.floor(k * (2 * x2 - 1)) * np.pi) / 2)))
        g = 1 + tmp2
        f1 = np.multiply(np.multiply(g, np.cos(0.5 * np.pi * x2)), np.cos(0.5 * np.pi * x1))
        f2 = np.multiply(np.multiply(g, np.sin(0.5 * np.pi * x2)), np.cos(0.5 * np.pi * x1))
        f3 = np.multiply(g, np.sin(0.5 * np.pi * x1))

        return get_PF(np.array([f1, f2, f3]), True)


class DF13(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_obj = 3
        self.xl[2:] = -1.0
        self.xu[2:] = +1.0

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.sin(0.5 * np.pi * self.time)
        p = np.floor(6 * G)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)
        g = 1 + np.sum((x[:, 2:] - G) ** 2, axis=1)
        g = g.reshape(len(g), 1)
        f1 = g * np.cos(0.5 * np.pi * x0) ** 2
        f2 = g * np.cos(0.5 * np.pi * x1) ** 2
        f3 = g * np.sin(0.5 * np.pi * x0) ** 2 + np.sin(0.5 * np.pi * x0) * np.cos(p * np.pi * x0) ** 2 + np.sin(
            0.5 * np.pi * x1) ** 2 + np.sin(0.5 * np.pi * x1) * np.cos(p * np.pi * x1) ** 2

        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = np.sin(0.5 * np.pi * self.time)
        p = np.floor(6 * G)

        f1 = np.cos(0.5 * np.pi * x1) ** 2
        f2 = np.cos(0.5 * np.pi * x2) ** 2
        f3 = np.sin(0.5 * np.pi * x1) ** 2 + np.sin(0.5 * np.pi * x1) * np.cos(p * np.pi * x1) ** 2 + np.sin(
            0.5 * np.pi * x2) ** 2 + \
             np.sin(0.5 * np.pi * x2) * np.cos(p * np.pi * x2) ** 2

        return get_PF(np.array([f1, f2, f3]), True)


class DF14(DF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_obj = 3
        self.xl[2:] = -1.0
        self.xu[2:] = +1.0

    def _evaluate(self, x, out, *args, **kwargs):
        G = np.sin(0.5 * np.pi * self.time)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)

        g = 1 + np.sum((x[:, 2:] - G) ** 2, axis=1)
        g = g.reshape(len(g), 1)
        y = 0.5 + G * (x0 - 0.5)

        f1 = g * (1 - y + 0.05 * np.sin(6 * np.pi * y))
        f2 = g * (1 - x1 + 0.05 * np.sin(6 * np.pi * x1)) * (y + 0.05 * np.sin(6 * np.pi * y))
        f3 = g * (x1 + 0.05 * np.sin(6 * np.pi * x1)) * (y + 0.05 * np.sin(6 * np.pi * y))

        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = np.sin(0.5 * np.pi * self.time)
        y = 0.5 + G * (x1 - 0.5)
        f1 = 1 - y + 0.05 * np.sin(6 * np.pi * y)
        f2 = np.multiply(1 - x2 + 0.05 * np.sin(6 * np.pi * x2), y + 0.05 * np.sin(6 * np.pi * y))
        f3 = np.multiply(x2 + 0.05 * np.sin(6 * np.pi * x2), y + 0.05 * np.sin(6 * np.pi * y))

        return get_PF(np.array([f1, f2, f3]), False)


def get_PF(f=None, nondominate=None):
    nds = NonDominatedSorting()
    ncell = len(f)
    s = np.size(f[1])
    h = []
    for i in np.arange(ncell):
        fi = np.reshape(f[i], s, order='F')
        h.append(fi)
    h = np.array(h).T
    h = np.reshape(h, (s, ncell))

    if nondominate:
        fronts = nds.do(F=h, only_non_dominated_front=True)
        h = h[fronts]
    return h
