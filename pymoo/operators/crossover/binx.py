import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.operators.crossover.util import row_at_least_once_true


def mut_binomial(n, m, prob, at_least_once=True):
    M = np.random.random((n, m)) < prob

    if at_least_once:
        M = row_at_least_once_true(M)

    return M


class BinomialCrossover(Crossover):

    def __init__(self, bias, **kwargs):
        super().__init__(2, 1, **kwargs)
        self.bias = bias

    def do(self, problem, pop, parents, **kwargs):
        X = pop.get("X")[parents.T].copy()

        _, n_matings, n_var = X.shape
        M = mut_binomial(n_matings, n_var, self.bias, at_least_once=True)

        Xp = X[0]
        Xp[~M] = X[1][~M]

        return Population.new(X=Xp)
