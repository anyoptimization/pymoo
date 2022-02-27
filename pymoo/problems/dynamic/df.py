import numpy as np

from pymoo.problems.dyn import DynamicTestProblem
from pymoo.util.remote import Remote


class DF(DynamicTestProblem):

    def __init__(self, n_var=10, nt=10, taut=20):
        super().__init__(nt,
                         taut,
                         n_var=n_var,
                         n_obj=2,
                         xl=0,
                         xu=1)

    def _calc_pareto_front(self, *args, **kwargs):
        return Remote.get_instance().load(f"pf", "DF", f"{str(self.__class__.__name__)}.pf")


class DF1(DF):

    def _evaluate(self, X, out, *args, **kwargs):
        v = np.sin(0.5 * np.pi * self.time())
        G = np.abs(v)
        H = 0.75 * v + 1.25
        g = 1 + np.sum((X[:, 1:] - G) ** 2, axis=1)

        f1 = X[:, 0]
        f2 = g * (1 - np.power((f1 / g), H))

        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 0.75 * np.sin(0.5 * np.pi * self.time()) + 1.25
        f1 = np.linspace(0, 1, n_pareto_points)
        return np.array([f1, 1 - np.power(f1, H)]).T


class DF2(DF1):
    pass


class DF3(DF1):
    pass


class DF4(DF1):
    pass


class DF5(DF1):
    pass


class DF6(DF1):
    pass


class DF7(DF1):
    pass


class DF8(DF1):
    pass


class DF9(DF1):
    pass


class DF10(DF1):
    pass


class DF11(DF1):
    pass


class DF12(DF1):
    pass


class DF13(DF1):
    pass


class DF14(DF1):
    pass
