import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.variable import Real, get
from pymoo.util.misc import row_at_least_once_true


def mut_binomial(n, m, prob, at_least_once=True):
    prob = np.ones(n) * prob
    M = np.random.random((n, m)) < prob[:, None]

    if at_least_once:
        M = row_at_least_once_true(M)

    return M


class BinomialCrossover(Crossover):

    def __init__(self, bias=0.5, n_offsprings=2, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)
        self.bias = Real(bias, bounds=(0.1, 0.9), strict=(0.0, 1.0))

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        bias = get(self.bias, size=n_matings)
        M = mut_binomial(n_matings, n_var, bias, at_least_once=True)

        if self.n_offsprings == 1:
            Xp = X[0].copy()
            Xp[~M] = X[1][~M]
            Xp = Xp[None, ...]
        elif self.n_offsprings == 2:
            Xp = np.copy(X)
            Xp[0][~M] = X[1][~M]
            Xp[1][~M] = X[0][~M]
        else:
            raise Exception

        return Xp


class BX(BinomialCrossover):
    pass
