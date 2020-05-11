import os

import numpy as np

from pymoo.model.problem import Problem

# Based on the C++ implementation by the Ma and Wang
# http://www.escience.cn/people/yongwang1/index.html
from pymoo.problems.util import load_pareto_front_from_file


class MW(Problem):
    def __init__(self, n_var, n_obj, n_constr, **kwargs):
        if 'xl' not in kwargs:
            kwargs['xl'] = 0
        if 'xu' not in kwargs:
            kwargs['xu'] = 1
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         type_var=np.double, **kwargs)

    @staticmethod
    def LA1(A, B, C, D, theta):
        return A * np.power(np.sin(B * np.pi * np.power(theta, C)), D)

    @staticmethod
    def LA2(A, B, C, D, theta):
        return A * np.power(np.sin(B * np.power(theta, C)), D)

    @staticmethod
    def LA3(A, B, C, D, theta):
        return A * np.power(np.cos(B * np.power(theta, C)), D)

    def g1(self, X):
        d = self.n_var
        n = d - self.n_obj

        z = np.power(X[:, self.n_obj - 1:], n)
        i = np.arange(self.n_obj - 1, d)

        exp = 1 - np.exp(-10.0 * (z - 0.5 - i / (2 * d)) * (z - 0.5 - i / (2 * d)))
        distance = 1 + exp.sum(axis=1)
        return distance

    def g2(self, X):
        d = self.n_var
        n = d

        i = np.arange(self.n_obj - 1, d)
        z = 1 - np.exp(-10.0 * (X[:, self.n_obj - 1:] - i / n) * (X[:, self.n_obj - 1:] - i / n))
        contrib = (0.1 / (n)) * z * z + 1.5 - 1.5 * np.cos(2 * np.pi * z)
        distance = 1 + contrib.sum(axis=1)
        return distance

    def g3(self, X):
        contrib = 2.0 * np.power(
            X[:, self.n_obj - 1:] + (X[:, self.n_obj - 2:-1] - 0.5) * (X[:, self.n_obj - 2:-1] - 0.5) - 1.0, 2.0)
        distance = 1 + contrib.sum(axis=1)
        return distance


class MW1(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 1)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)
        f0 = X[:, 0]
        f1 = g * (1 - 0.85 * f0 / g)

        g0 = f0 + f1 - 1 - self.LA1(0.5, 2.0, 1.0, 8.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        out["F"] = np.column_stack([f0, f1])
        out["G"] = g0.reshape((-1, 1))

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            F = np.zeros((100, 2))
            F[:, 0] = np.linspace(0, 1, 100)
        else:
            F = ref_dirs
        F[:, 1] = 1 - 0.85 * F[:, 0]
        l = np.sqrt(2) * F[:, 1] - np.sqrt(2) * F[:, 0]
        c = 1 - F[:, 0] - F[:, 1] + 0.5 * np.sin(2 * np.pi * l) ** 8
        F = F[c >= 0]
        return F


class MW2(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 1)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)
        f0 = X[:, 0]
        f1 = g * (1 - f0 / g)

        g0 = f0 + f1 - 1 - self.LA1(0.5, 3.0, 1.0, 8.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        out["F"] = np.column_stack([f0, f1])
        out["G"] = g0.reshape((-1, 1))

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            F = np.zeros((100, 2))
            F[:, 0] = np.linspace(0, 1, 100)
        else:
            F = ref_dirs
        F[:, 1] = 1 - F[:, 0]
        return F


class MW3(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 2)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g3(X)
        f0 = X[:, 0]
        f1 = g * (1 - f0 / g)

        g0 = f0 + f1 - 1.05 - self.LA1(0.45, 0.75, 1.0, 6.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        g1 = 0.85 - f0 - f1 + self.LA1(0.3, 0.75, 1.0, 2.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        out["F"] = np.column_stack([f0, f1])
        out["G"] = np.column_stack([g0, g1])

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            F = np.zeros((100, 2))
            F[:, 0] = np.linspace(0, 1, 100)
        else:
            F = ref_dirs
        F[:, 1] = 1 - F[:, 0]
        invalid = (0.85 - F[:, 0] - F[:, 1] + 0.3 * np.sin(0.75 * np.pi * np.sqrt(2) * (F[:, 1] - F[:, 0])) ** 2) > 0
        while invalid.any():
            F[invalid, :] *= 1.001
            invalid = (0.85 - F[:, 0] - F[:, 1] + 0.3 * np.sin(
                0.75 * np.pi * np.sqrt(2) * (F[:, 1] - F[:, 0])) ** 2) > 0
        return F


class MW4(MW):
    def __init__(self, n_var=None, n_obj=3, **kwargs):
        if n_var is None:
            n_var = n_obj + 12
        super().__init__(n_var, n_obj, 1)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)
        f = g.reshape((-1, 1)) * np.ones((X.shape[0], self.n_obj))
        f[:, 1:] *= X[:, (self.n_obj - 2)::-1]
        f[:, 0:-1] *= np.flip(np.cumprod(1 - X[:, :(self.n_obj - 1)], axis=1), axis=1)

        g0 = f.sum(axis=1) - 1 - self.LA1(0.4, 2.5, 1.0, 8.0, f[:, -1] - f[:, :-1].sum(axis=1))
        out["F"] = f
        out["G"] = g0.reshape((-1, 1))

    def _calc_pareto_front(self, ref_dirs):
        F = ref_dirs
        l = F[:, -1] - np.sum(F[:, :-1], axis=1)
        c = (1 + 0.4 * np.sin(2.5 * np.pi * l) ** 8) - np.sum(F, axis=1)
        return F[c >= 0]


class MW5(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 3)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)
        f0 = g * X[:, 0]
        f1 = g * np.sqrt(1.0 - np.power(f0 / g, 2.0))

        with np.errstate(divide='ignore'):
            atan = np.arctan(f1 / f0)

        g0 = f0 ** 2 + f1 ** 2 - np.power(1.7 - self.LA2(0.2, 2.0, 1.0, 1.0, atan), 2.0)
        t = 0.5 * np.pi - 2 * np.abs(atan - 0.25 * np.pi)
        g1 = np.power(1 + self.LA2(0.5, 6.0, 3.0, 1.0, t), 2.0) - f0 ** 2 - f1 ** 2
        g2 = np.power(1 - self.LA2(0.45, 6.0, 3.0, 1.0, t), 2.0) - f0 ** 2 - f1 ** 2
        out["F"] = np.column_stack([f0, f1])
        out["G"] = np.column_stack([g0, g1, g2])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MW", "MW5.pf"))


class MW6(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 1, xl=0.0, xu=1.1)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)
        f0 = g * X[:, 0]
        f1 = g * np.sqrt(1.1 * 1.1 - np.power(f0 / g, 2.0))

        with np.errstate(divide='ignore'):
            atan = np.arctan(f1 / f0)

        g0 = f0 ** 2 / np.power(1.0 + self.LA3(0.15, 6.0, 4.0, 10.0, atan), 2.0) + f1 ** 2 / np.power(
            1.0 + self.LA3(0.75, 6.0, 4.0, 10.0, atan), 2.0) - 1
        out["F"] = np.column_stack([f0, f1])
        out["G"] = g0.reshape((-1, 1))

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            F = np.zeros((100, 2))
            F[:, 0] = np.linspace(0, 1, 100)
        else:
            F = ref_dirs
        F[:, 1] = 1 - F[:, 0]
        F = F / np.sqrt(np.sum(F ** 2, axis=1) / 1.21).reshape((-1, 1))
        l = np.cos(6 * np.arctan(F[:, 1] / F[:, 0]) ** 4) ** 10
        c = 1 - (F[:, 0] / (1 + 0.15 * l)) ** 2 - (F[:, 1] / (1 + 0.75 * l)) ** 2
        return F[c >= 0]


class MW7(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 2)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g3(X)
        f0 = g * X[:, 0]
        f1 = g * np.sqrt(1 - np.power(f0 / g, 2))

        with np.errstate(divide='ignore'):
            atan = np.arctan(f1 / f0)

        g0 = f0 ** 2 + f1 ** 2 - np.power(1.2 + np.abs(self.LA2(0.4, 4.0, 1.0, 16.0, atan)), 2.0)
        g1 = np.power(1.15 - self.LA2(0.2, 4.0, 1.0, 8.0, atan), 2.0) - f0 ** 2 - f1 ** 2
        out["F"] = np.column_stack([f0, f1])
        out["G"] = np.column_stack([g0, g1])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MW", "MW7.pf"))


class MW8(MW):
    def __init__(self, n_var=None, n_obj=3, **kwargs):
        if n_var is None:
            n_var = n_obj + 12
        super().__init__(n_var, n_obj, 1)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)
        f = g.reshape((-1, 1)) * np.ones((X.shape[0], self.n_obj))
        f[:, 1:] *= np.sin(0.5 * np.pi * X[:, (self.n_obj - 2)::-1])
        cos = np.cos(0.5 * np.pi * X[:, :(self.n_obj - 1)])
        f[:, 0:-1] *= np.flip(np.cumprod(cos, axis=1), axis=1)

        f_squared = (f ** 2).sum(axis=1)
        g0 = f_squared - (1.25 - self.LA2(0.5, 6.0, 1.0, 2.0, np.arcsin(f[:, -1] / np.sqrt(f_squared)))) * (
                1.25 - self.LA2(0.5, 6.0, 1.0, 2.0, np.arcsin(f[:, -1] / np.sqrt(f_squared))))
        out["F"] = f
        out["G"] = g0.reshape((-1, 1))

    def _calc_pareto_front(self, ref_dirs):
        F = ref_dirs
        F = F / np.sqrt(np.sum(F ** 2, axis=1)).reshape((-1, 1))
        c = (1.25 - 0.5 * np.sin(6 * np.arcsin(F[:, -1])) ** 2) ** 2 - np.sum(F ** 2, axis=1)
        return F[c >= 0]


class MW9(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 1)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)
        f0 = g * X[:, 0]
        f1 = g * (1.0 - np.power(f0 / g, 0.6))

        t1 = (1 - 0.64 * f0 * f0 - f1) * (1 - 0.36 * f0 * f0 - f1)
        t2 = (1.35 * 1.35 - (f0 + 0.35) * (f0 + 0.35) - f1) * (1.15 * 1.15 - (f0 + 0.15) * (f0 + 0.15) - f1)
        g0 = np.minimum(t1, t2)
        out["F"] = np.column_stack([f0, f1])
        out["G"] = g0.reshape((-1, 1))

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MW", "MW9.pf"))


class MW10(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 3)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)
        f0 = g * np.power(X[:, 0], self.n_var)
        f1 = g * (1.0 - np.power(f0 / g, 2.0))

        g0 = -1.0 * (2.0 - 4.0 * f0 * f0 - f1) * (2.0 - 8.0 * f0 * f0 - f1)
        g1 = (2.0 - 2.0 * f0 * f0 - f1) * (2.0 - 16.0 * f0 * f0 - f1)
        g2 = (1.0 - f0 * f0 - f1) * (1.2 - 1.2 * f0 * f0 - f1)
        out["F"] = np.column_stack([f0, f1])
        out["G"] = np.column_stack([g0, g1, g2])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MW", "MW10.pf"))


class MW11(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 4, xl=0.0, xu=np.sqrt(2))

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g3(X)
        f0 = g * X[:, 0]
        f1 = g * np.sqrt(2.0 - np.power(f0 / g, 2.0))

        g0 = -1.0 * (3.0 - f0 * f0 - f1) * (3.0 - 2.0 * f0 * f0 - f1)
        g1 = (3.0 - 0.625 * f0 * f0 - f1) * (3.0 - 7.0 * f0 * f0 - f1)
        g2 = -1.0 * (1.62 - 0.18 * f0 * f0 - f1) * (1.125 - 0.125 * f0 * f0 - f1)
        g3 = (2.07 - 0.23 * f0 * f0 - f1) * (0.63 - 0.07 * f0 * f0 - f1)
        out["F"] = np.column_stack([f0, f1])
        out["G"] = np.column_stack([g0, g1, g2, g3])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MW", "MW11.pf"))


class MW12(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 2)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)
        f0 = g * X[:, 0]
        f1 = g * (0.85 - 0.8 * (f0 / g) - 0.08 * np.abs(np.sin(3.2 * np.pi * (f0 / g))))

        g0 = -1.0 * (1 - 0.625 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 - f0 / 1.6))) * (
                1.4 - 0.875 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 / 1.4 - f0 / 1.6)))
        g1 = (1 - 0.8 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 - f0 / 1.5))) * (
                1.8 - 1.125 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 / 1.8 - f0 / 1.6)))
        out["F"] = np.column_stack([f0, f1])
        out["G"] = np.column_stack([g0, g1])

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            F = np.zeros((100, 2))
            F[:, 0] = np.linspace(0, 1, 100)
        else:
            F = ref_dirs
        F[:, 1] = 0.85 - 0.8 * F[:, 0] - 0.08 * np.abs(np.sin(3.2 * np.pi * F[:, 0]))

        invalid = (1 - 0.8 * F[:, 0] - F[:, 1] + 0.08 * np.sin(2 * np.pi * (F[:, 1] - F[:, 0] / 1.5))) * (
                1.8 - 1.125 * F[:, 0] - F[:, 1] + 0.08 * np.sin(2 * np.pi * (F[:, 1] / 1.8 - F[:, 0] / 1.6))) > 0
        while invalid.any():
            F[invalid, :] *= 1.001
            invalid = (1 - 0.8 * F[:, 0] - F[:, 1] + 0.08 * np.sin(2 * np.pi * (F[:, 1] - F[:, 0] / 1.5))) * (
                    1.8 - 1.125 * F[:, 0] - F[:, 1] + 0.08 * np.sin(
                2 * np.pi * (F[:, 1] / 1.8 - F[:, 0] / 1.6))) > 0
        return F


class MW13(MW):
    def __init__(self, n_var=15, **kwargs):
        super().__init__(n_var, 2, 2, xu=1.5)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)
        f0 = g * X[:, 0]
        f1 = g * (5.0 - np.exp(f0 / g) - np.abs(0.5 * np.sin(3 * np.pi * f0 / g)))

        g0 = -1.0 * (5.0 - (1 + f0 + 0.5 * f0 * f0) - 0.5 * np.sin(3 * np.pi * f0) - f1) * (
                5.0 - (1 + 0.7 * f0) - 0.5 * np.sin(3 * np.pi * f0) - f1)
        g1 = (5.0 - np.exp(f0) - 0.5 * np.sin(3 * np.pi * f0) - f1) * (
                5.0 - (1 + 0.4 * f0) - 0.5 * np.sin(3 * np.pi * f0) - f1)
        out["F"] = np.column_stack([f0, f1])
        out["G"] = np.column_stack([g0, g1])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MW", "MW13.pf"))


class MW14(MW):
    def __init__(self, n_var=None, n_obj=3, **kwargs):
        if n_var is None:
            n_var = n_obj + 12
        super().__init__(n_var, n_obj, 1, xu=1.5)

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g3(X)
        f = np.zeros((X.shape[0], self.n_obj))
        f[:, :-1] = X[:, :(self.n_obj - 1)]
        LA1 = self.LA1(1.5, 1.1, 2.0, 1.0, f[:, :-1])
        inter = (6 - np.exp(f[:, :-1]) - LA1).sum(axis=1)
        f[:, -1] = g / (self.n_obj - 1) * inter

        alpha = 6.1 - 1 - f[:, :-1] - 0.5 * f[:, :-1] * f[:, :-1] - LA1
        g0 = f[:, -1] - 1 / (self.n_obj - 1) * alpha.sum(axis=1)
        out["F"] = f
        out["G"] = g0.reshape((-1, 1))

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MW", "MW14.pf"))
