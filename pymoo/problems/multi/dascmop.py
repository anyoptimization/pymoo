import os

import numpy as np

from pymoo.model.problem import Problem
from pymoo.problems.util import load_pareto_front_from_file

DIFFICULTIES = [
    (0.25, 0., 0.), (0., 0.25, 0.), (0., 0., 0.25), (0.25, 0.25, 0.25),
    (0.5, 0., 0.), (0., 0.5, 0.), (0., 0., 0.5), (0.5, 0.5, 0.5),
    (0.75, 0., 0.), (0., 0.75, 0.), (0., 0., 0.75), (0.75, 0.75, 0.75),
    (0., 1.0, 0.), (0.5, 1.0, 0.), (0., 1.0, 0.5), (0.5, 1.0, 0.5)
]


class DASCMOP(Problem):
    def __init__(self, n_obj, n_constr, difficulty, **kwargs):
        super().__init__(n_var=30,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         type_var=np.double, xl=0., xu=1., **kwargs)


        if isinstance(difficulty, int):
            self.difficulty = difficulty
            if not (1 <= difficulty <= len(DIFFICULTIES)):
                raise Exception(f"Difficulty must be 1 <= difficulty <= {len(DIFFICULTIES)}, but it is {difficulty}!")
            vals = DIFFICULTIES[difficulty-1]
        else:
            self.difficulty = -1
            vals = difficulty

        self.eta, self.zeta, self.gamma = vals

    def g1(self, X):
        contrib = (X[:, self.n_obj - 1:] - np.sin(0.5 * np.pi * X[:, 0:1])) ** 2
        return contrib.sum(axis=1)[:, None]

    def g2(self, X):
        z = X[:, self.n_obj - 1:] - 0.5
        contrib = z ** 2 - np.cos(20 * np.pi * z)
        return (self.n_var - self.n_obj + 1) + contrib.sum(axis=1)[:, None]

    def g3(self, X):
        j = np.arange(self.n_obj - 1, self.n_var) + 1
        contrib = (X[:, self.n_obj - 1:] - np.cos(0.25 * j / self.n_var * np.pi * (X[:, 0:1] + X[:, 1:2]))) ** 2
        return contrib.sum(axis=1)[:, None]

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{str(self.__class__.__name__).lower()}_{self.difficulty}.pf"
        return load_pareto_front_from_file(os.path.join("DASCMOP", fname))

class DASCMOP1(DASCMOP):
    def __init__(self, difficulty, **kwargs):
        super().__init__(2, 11, difficulty)

    def constraints(self, X, f0, f1, g):
        a = 20.
        b = 2. * self.eta - 1.
        d = 0.5 if self.zeta != 0 else 0.
        if self.zeta > 0:
            e = d - np.log(self.zeta)
        else:
            e = 1e30
        r = 0.5 * self.gamma

        p_k = np.array([[0., 1.0, 0., 1.0, 2.0, 0., 1.0, 2.0, 3.0]])
        q_k = np.array([[1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5]])

        a_k2 = 0.3
        b_k2 = 1.2
        theta_k = -0.25 * np.pi

        c = np.zeros((X.shape[0], 2 + p_k.shape[1]))

        c[:, 0] = np.sin(a * np.pi * X[:, 0]) - b
        if self.zeta == 1.:
            c[:, 1:2] = 1e-4 - np.abs(e - g)
        else:
            c[:, 1:2] = (e - g) * (g - d)

        c[:, 2:] = (((f0 - p_k) * np.cos(theta_k) - (f1 - q_k) * np.sin(theta_k)) ** 2 / a_k2
                    + ((f0 - p_k) * np.sin(theta_k) + (f1 - q_k) * np.cos(theta_k)) ** 2 / b_k2
                    - r)

        return -1 * c

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)

        f0 = X[:, 0:1] + g
        f1 = 1.0 - X[:, 0:1] ** 2 + g

        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP2(DASCMOP1):
    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)

        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + g

        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP3(DASCMOP1):
    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g1(X)

        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + 0.5 * np.abs(np.sin(5 * np.pi * X[:, 0:1])) + g

        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP4(DASCMOP1):
    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)

        f0 = X[:, 0:1] + g
        f1 = 1.0 - X[:, 0:1] ** 2 + g

        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP5(DASCMOP1):

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)

        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + g

        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP6(DASCMOP1):
    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)

        f0 = X[:, 0:1] + g
        f1 = 1.0 - np.sqrt(X[:, 0:1]) + 0.5 * np.abs(np.sin(5 * np.pi * X[:, 0:1])) + g

        out["F"] = np.column_stack([f0, f1])
        out["G"] = self.constraints(X, f0, f1, g)


class DASCMOP7(DASCMOP):

    def __init__(self, difficulty_factors, **kwargs):
        super().__init__(3, 7, difficulty_factors)

    def constraints(self, X, f0, f1, f2, g):
        a = 20.
        b = 2. * self.eta - 1.
        d = 0.5 if self.zeta != 0 else 0
        if self.zeta > 0:
            e = d - np.log(self.zeta)
        else:
            e = 1e30
        r = 0.5 * self.gamma

        x_k = np.array([[1.0, 0., 0., 1.0 / np.sqrt(3.0)]])
        y_k = np.array([[0., 1.0, 0., 1.0 / np.sqrt(3.0)]])
        z_k = np.array([[0., 0., 1.0, 1.0 / np.sqrt(3.0)]])

        c = np.zeros((X.shape[0], 3 + x_k.shape[1]))

        c[:, 0] = np.sin(a * np.pi * X[:, 0]) - b
        c[:, 1] = np.cos(a * np.pi * X[:, 1]) - b
        if self.zeta == 1:
            c[:, 2:3] = 1e-4 - np.abs(e - g)
        else:
            c[:, 2:3] = (e - g) * (g - d)

        c[:, 3:] = (f0 - x_k) ** 2 + (f1 - y_k) ** 2 + (f2 - z_k) ** 2 - r ** 2
        return -1 * c

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)

        f0 = X[:, 0:1] * X[:, 1:2] + g
        f1 = X[:, 1:2] * (1.0 - X[:, 0:1]) + g
        f2 = 1 - X[:, 1:2] + g

        out["F"] = np.column_stack([f0, f1, f2])
        out["G"] = self.constraints(X, f0, f1, f2, g)


class DASCMOP8(DASCMOP7):

    def objectives(self, X, g):
        f0 = np.cos(0.5 * np.pi * X[:, 0:1]) * np.cos(0.5 * np.pi * X[:, 1:2]) + g
        f1 = np.cos(0.5 * np.pi * X[:, 0:1]) * np.sin(0.5 * np.pi * X[:, 1:2]) + g
        f2 = np.sin(0.5 * np.pi * X[:, 0:1]) + g
        return np.column_stack([f0, f1, f2])

    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g2(X)
        F = self.objectives(X, g)
        out["F"] = F
        out["G"] = self.constraints(X, F[:, 0:1], F[:, 1:2], F[:, 2:3], g)


class DASCMOP9(DASCMOP8):
    def _evaluate(self, X, out, *args, **kwargs):
        g = self.g3(X)
        F = self.objectives(X, g)
        out["F"] = F
        out["G"] = self.constraints(X, F[:, 0:1], F[:, 1:2], F[:, 2:3], g)
