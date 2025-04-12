from pymoo.core.problem import Problem


class Himmelblau(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, xl=-6, xu=6, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
