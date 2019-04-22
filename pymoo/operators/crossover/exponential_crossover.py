import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask
from pymoo.rand import random


class ExponentialCrossover(Crossover):

    def __init__(self, prob):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, pop, parents, **kwargs):

        # get the X of parents and count the matings
        X = pop.get("X")[parents.T]
        _, n_matings, n_var = X.shape

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # start point of crossover
        n = random.randint(0, n_var, size=len(pop))

        # the probabilities are calculated beforehand
        r = random.random((n_matings, n_var)) < self.prob

        # create for each individual the crossover range
        for i in range(n_matings):

            # the actual index where we start
            start = n[i]
            for j in range(problem.n_var):

                # the current position where we are pointing to
                current = (start + j) % problem.n_var

                # replace only if random value keeps being smaller than CR
                if r[i, current]:
                    M[i, current] = True
                else:
                    break

        _X = crossover_mask(X, M)
        return pop.new("X", _X)
