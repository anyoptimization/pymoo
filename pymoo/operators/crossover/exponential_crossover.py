import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossver_by_mask
from pymoo.rand import random


class ExponentialCrossover(Crossover):

    def __init__(self, prob):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, pop, parents, **kwargs):

        # get the X of parents and count the matings
        X = pop.get("X")[parents.T]
        n_matings = parents.shape[0]

        # the mask do to the crossover
        M = np.full((len(pop), problem.n_var), False)

        # start point of crossover
        n = random.randint(0, problem.n_var, size=len(pop))

        # the probabilities are calculated beforehand
        r = np.random((n_matings, problem.n_var)) < self.prob

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

        _X = crossver_by_mask(X, M)
        return pop.new("X", _X)
