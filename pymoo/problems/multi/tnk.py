import autograd.numpy as anp

from pymoo.problems.util import load_pareto_front_from_file
from pymoo.model.problem import Problem


class TNK(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, type_var=anp.double)
        self.xl = anp.array([0, 1e-30])
        self.xu = anp.array([anp.pi, anp.pi])

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        f2 = x[:, 1]
        g1 = -(anp.square(x[:, 0]) + anp.square(x[:, 1]) - 1.0 - 0.1 * anp.cos(16.0 * anp.arctan(x[:, 0] / x[:, 1])))
        g2 = 2 * (anp.square(x[:, 0] - 0.5) + anp.square(x[:, 1] - 0.5)) - 1

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_front(self, *args, **kwargs):
        return load_pareto_front_from_file("tnk.pf")
