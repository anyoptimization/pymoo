import autograd.numpy as anp

from pymoo.problems.util import load_pareto_front_from_file
from pymoo.model.problem import Problem


class Kursawe(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=-5, xu=5, type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        l = []
        for i in range(2):
            l.append(-10 * anp.exp(-0.2 * anp.sqrt(anp.square(x[:, i]) + anp.square(x[:, i + 1]))))
        f1 = anp.sum(anp.column_stack(l), axis=1)

        f2 = anp.sum(anp.power(anp.abs(x), 0.8) + 5 * anp.sin(anp.power(x, 3)), axis=1)

        out["F"] = anp.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, **kwargs):
        return load_pareto_front_from_file("kursawe.pf")
