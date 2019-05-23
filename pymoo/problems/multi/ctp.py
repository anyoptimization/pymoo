import autograd.numpy as anp

from pymoo.model.problem import Problem
from pymoo.problems.util import load_pareto_front_from_file


class CTP(Problem):

    def __init__(self, n_var=2, n_constr=1, option="linear"):
        super().__init__(n_var=n_var, n_obj=2, n_constr=n_constr, xl=0, xu=1, type_var=anp.double)

        def g_linear(x):
            return 1 + anp.sum(x, axis=1)

        def g_multimodal(x):
            A = 10
            return 1 + A * x.shape[1] + anp.sum(x ** 2 - A * anp.cos(2 * anp.pi * x), axis=1)

        if option == "linear":
            self.calc_g = g_linear

        elif option == "multimodal":
            self.calc_g = g_multimodal
            self.xl[:, 1:] = -5.12
            self.xu[:, 1:] = 5.12

        else:
            print("Unknown option for CTP single.")

    def calc_objectives(self, x):
        f1 = x[:, 0]
        gg = self.calc_g(x[:, 1:])
        f2 = gg * (1 - anp.sqrt(f1 / gg))
        return f1, f2

    def calc_constraint(self, theta, a, b, c, d, e, f1, f2):
        return - (anp.cos(theta) * (f2 - e) - anp.sin(theta) * f1 -
                  a * anp.abs(anp.sin(b * anp.pi * (anp.sin(theta) * (f2 - e) + anp.cos(theta) * f1) ** c)) ** d)


class CTP1(CTP):

    def __init__(self, n_var=2, n_constr=2, **kwargs):
        super().__init__(n_var, n_constr, **kwargs)

        a, b = anp.zeros(n_constr + 1), anp.zeros(n_constr + 1)
        a[0], b[0] = 1, 1
        delta = 1 / (n_constr + 1)
        alpha = delta

        for j in range(n_constr):
            beta = a[j] * anp.exp(-b[j] * alpha)
            a[j + 1] = (a[j] + beta) / 2
            b[j + 1] = - 1 / alpha * anp.log(beta / a[j + 1])

            alpha += delta

        self.a = a[1:]
        self.b = b[1:]

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp1.pf")

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        gg = self.calc_g(x[:, 1:])
        f2 = gg * anp.exp(-f1 / gg)
        out["F"] = anp.column_stack([f1, f2])

        a, b = self.a, self.b
        g = []
        for j in range(self.n_constr):
            _g = - (f2 - (a[j] * anp.exp(-b[j] * f1)))
            g.append(_g)
        out["G"] = anp.column_stack(g)


class CTP2(CTP):

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp2.pf")

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.2, 10, 1, 6, 1
        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP3(CTP):

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp3.pf")

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.1, 10, 1, 0.5, 1

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP4(CTP):

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp4.pf")

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.75, 10, 1, 0.5, 1

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP5(CTP):

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp5.pf")

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.2 * anp.pi
        a, b, c, d, e = 0.1, 10, 2, 0.5, 1

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP6(CTP):

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp6.pf")

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = 0.1 * anp.pi
        a, b, c, d, e = 40, 0.5, 1, 2, -2

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP7(CTP):

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp7.pf")

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = self.calc_objectives(x)
        out["F"] = anp.column_stack([f1, f2])

        theta = -0.05 * anp.pi
        a, b, c, d, e = 40, 5, 1, 6, 0

        out["G"] = self.calc_constraint(theta, a, b, c, d, e, f1, f2)


class CTP8(CTP):
    def __init__(self, **kwargs):
        super().__init__(n_constr=2, **kwargs)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("ctp8.pf")

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
    problem = CTP1(n_constr=3)
    print(problem.n_constr)
