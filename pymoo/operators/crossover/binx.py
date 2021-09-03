import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.model.population import Population


def mut_biased(n, m, prob, at_least_once=True):
    M = np.random.random((n, m)) < prob

    if at_least_once:
        for k in np.where(~np.any(M, axis=1))[0]:
            M[k, np.random.randint(m)] = True

    return M


class BiasedCrossover(Crossover):

    def __init__(self, bias, **kwargs):
        super().__init__(2, 1, **kwargs)
        self.bias = bias

    def do(self, problem, pop, parents, **kwargs):
        X = pop.get("X")[parents.T].copy()

        _, n_matings, n_var = X.shape
        M = mut_biased(n_matings, n_var, self.bias, at_least_once=True)

        Xp = X[0]
        Xp[~M] = X[1][~M]

        return Population.new(X=Xp)
