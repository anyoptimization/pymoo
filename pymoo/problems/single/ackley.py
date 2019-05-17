import autograd.numpy as np

from pymoo.model.problem import Problem


class Ackley(Problem):

    def __init__(self, n_var=10, c1=20, c2=.2, c3=2 * np.pi):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-32, xu=32, type_var=np.double)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = -1. * self.c1 * np.exp(-1. * self.c2 * np.sqrt((1. / self.n_var) * np.sum(x * x, axis=1)))
        part2 = -1. * np.exp((1. / self.n_var) * np.sum(np.cos(self.c3 * x), axis=1))
        out["F"] = part1 + part2 + self.c1 + np.exp(1)

    def _calc_pareto_front(self):
        return np.zeros((1, 1))
