import autograd.numpy as np


from pymoo.core.problem import Problem
from pymoo.problems.autodiff import AutomaticDifferentiation
from pymoo.problems.multi import ZDT, ZDT1


class ElementwiseZDT1(Problem):

    def __init__(self, n_var=30, n_obj=2, n_constr=0, **kwargs):
        super().__init__(n_var, evaluation_of=["F", "dF"], elementwise_evaluation=True, n_obj=n_obj, n_constr=n_constr,
                         **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]
        g = 1 + 9.0 / (self.n_var - 1) * x[1:].sum()
        f2 = g * (1 - (f1 / g) ** 0.5)
        out["F"] = ([f1, f2])


class MySphere(Problem):

    def __init__(self, gamma=1):
        super().__init__(n_var=2, n_obj=1, n_constr=1, elementwise_evaluation=True,
                         evaluation_of=["F", "dF", "dG", "ddF", "ddG"])
        self.gamma = gamma

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 3).sum()
        out["G"] = 1 + (x ** 2).sum()


class ZDT1WithGradient(ZDT):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var, evaluation_of=["F", "dF"], **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out["F"] = np.column_stack([f1, f2])

        if "dF" in out:
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=float)
            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -0.5 * np.sqrt(g / x[:, 0])
            dF[:, 1, 1:] = ((9 / (self.n_var - 1)) * (1 - 0.5 * np.sqrt(x[:, 0] / g)))[:, None]
            out["dF"] = dF


if __name__ == '__main__':
    # problem = MySphere()
    # problem = ZDT1WithGradient()

    problem = AutomaticDifferentiation(ZDT1())

    X = np.random.random((10, problem.n_var))

    F, dF = problem.evaluate(X, return_values_of=["F", "dF"])

    print(dF)
