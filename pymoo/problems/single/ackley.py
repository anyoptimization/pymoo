import autograd.numpy as np

from pymoo.model.problem import Problem


class Ackley(Problem):

    def __init__(self, n_var=10, a=20, b=1/5, c=2 * np.pi):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=-32.768, xu=+32.768, type_var=np.double)
        self.a = a
        self.b = b
        self.c = c

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = -1. * self.a * np.exp(-1. * self.b * np.sqrt((1. / self.n_var) * np.sum(x * x, axis=1)))
        part2 = -1. * np.exp((1. / self.n_var) * np.sum(np.cos(self.c * x), axis=1))
        out["F"] = part1 + part2 + self.a + np.exp(1)

    def _calc_pareto_front(self):
        return np.zeros((1, 1))
