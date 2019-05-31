import autograd.numpy as anp

from pymoo.problems.util import load_pareto_front_from_file
from pymoo.model.problem import Problem


class OSY(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=2, n_constr=6, type_var=anp.double)
        self.xl = anp.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.xu = anp.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0])

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = - (25 * anp.square(x[:, 0] - 2) + anp.square(x[:, 1] - 2) + anp.square(x[:, 2] - 1) + anp.square(x[:, 3] - 4) + anp.square(x[:, 4] - 1))

        f2 = anp.sum(anp.square(x), axis=1)

        g1 = (x[:, 0] + x[:, 1] - 2.0) / 2.0
        g2 = (6.0 - x[:, 0] - x[:, 1]) / 6.0
        g3 = (2.0 - x[:, 1] + x[:, 0]) / 2.0
        g4 = (2.0 - x[:, 0] + 3.0 * x[:, 1]) / 2.0
        g5 = (4.0 - anp.square(x[:, 2] - 3.0) - x[:, 3]) / 4.0
        g6 = (anp.square(x[:, 4] - 3.0) + x[:, 5] - 4.0) / 4.0

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = - anp.column_stack([g1, g2, g3, g4, g5, g6])

    def _calc_pareto_front(self):
        return load_pareto_front_from_file("osy.pf")

