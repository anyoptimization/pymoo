import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote


def g_linear(x):
    return 1 + np.sum(x, axis=1)


def g_multimodal(x):
    A = 10
    return 1 + A * x.shape[1] + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=1)


class CTP(Problem):

    def __init__(self, n_var=2, n_ieq_constr=1, option="linear"):
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=n_ieq_constr, xl=0, xu=1, vtype=float)

        if option == "linear":
            self.calc_g = g_linear

        elif option == "multimodal":
            self.calc_g = g_multimodal
            self.xl[1:] = -5.12
            self.xu[1:] = 5.12

        else:
            print("Unknown option for CTP single.")

    def calc_objectives(self, x):
        f1 = x[:, 0]
        gg = self.calc_g(x[:, 1:])
        f2 = gg * (1 - (f1 / gg) ** 0.5)
        return f1, f2

    def calc_constraint(self, theta, a, b, c, d, e, f1, f2):

        # Equations in readable format
        exp1 = (f2 - e) * anp.cos(theta) - f1 * anp.sin(theta)

        exp2 = (f2 - e) * anp.sin(theta) + f1 * anp.cos(theta)
        exp2 = b * anp.pi * (exp2 ** c)
        exp2 = anp.abs(anp.sin(exp2))
        exp2 = a * (exp2 ** d)

        # as in the paper
        # val = - (exp1 - exp2)

        # as in the C code of NSGA2
        val = 1 - exp1 / exp2

        # ONE EQUATION
        # _val = - (anp.cos(theta) * (f2 - e) - anp.sin(theta) * f1 -
        #           a * anp.abs(anp.sin(b * anp.pi * (anp.sin(theta) * (f2 - e) + anp.cos(theta) * f1) ** c)) ** d)

        return val

    def _calc_pareto_front(self, *args, **kwargs):
        return Remote.get_instance().load("pymoo", "pf", "CTP", str(self.__class__.__name__).lower() + ".pf")


class CTP1(CTP):

    def __init__(self, n_var=2, n_ieq_constr=2, **kwargs):
        super().__init__(n_var, n_ieq_constr, **kwargs)

        a, b = np.zeros(n_ieq_constr + 1), np.zeros(n_ieq_constr + 1)
        a[0], b[0] = 1, 1
        delta = 1 / (n_ieq_constr + 1)
        alpha = delta

        for j in range(n_ieq_constr):
            beta = a[j] * np.exp(-b[j] * alpha)
            a[j + 1] = (a[j] + beta) / 2
            b[j + 1] = - 1 / alpha * np.log(beta / a[j + 1])

            alpha += delta

        self.a = a[1:]
        self.b = b[1:]

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        gg = self.calc_g(x[:, 1:])
        f2 = gg * anp.exp(-f1 / gg)
        out["F"] = anp.column_stack([f1, f2])

        a, b = self.a, self.b
        g = []
        for j in range(self.n_ieq_constr):
            _g = - (f2 - (a[j] * anp.exp(-b[j] * f1)))
            g.append(_g)
        out["G"] = anp.column_stack(g)


class CTP2(CTP):

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.2, 10, 1, 6, 1
        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP3(CTP):

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.1, 10, 1, 0.5, 1

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP4(CTP):

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.75, 10, 1, 0.5, 1

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP5(CTP):

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.1, 10, 2, 0.5, 1

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP6(CTP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xu = np.full(self.n_var, 20)
        self.xu[0] = 1

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = 0.1 * anp.pi
        a, b, c, d, e = 40, 0.5, 1, 2, -2

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP7(CTP):

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.05 * anp.pi
        a, b, c, d, e = 40, 5, 1, 6, 0

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP8(CTP):
    def __init__(self, **kwargs):
        super().__init__(n_ieq_constr=2, **kwargs)
        self.xu = np.full(self.n_var, 20)
        self.xu[0] = 1

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = 0.1 * anp.pi
        a, b, c, d, e = 40, 0.5, 1, 2, -2
        g1 = self.calc_constraint(theta, a, b, c, d, e, f1, f2)

        theta = -0.05 * anp.pi
        a, b, c, d, e = 40, 2, 1, 6, 0
        g2 = self.calc_constraint(theta, a, b, c, d, e, f1, f2)

        out["G"] = anp.column_stack([g1, g2])


if __name__ == '__main__':
    problem = CTP1(n_ieq_constr=3)
    print(problem.n_ieq_constr)
