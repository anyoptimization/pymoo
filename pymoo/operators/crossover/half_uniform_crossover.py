from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossver_by_mask

import numpy as np

from pymoo.rand import random


class HalfUniformCrossover(Crossover):

    def __init__(self, prob=0.5):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, pop, parents, **kwargs):

        # get the X of parents and count the matings
        X = pop.get("X")[parents.T]
        _, n_matings, n_var = X.shape

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        not_equal = X[0] != X[1]

        # create for each individual the crossover range
        for i in range(n_matings):

            I = np.where(not_equal[i])[0]

            if len(I) == 0:
                pass
            elif len(I) == 1:
                M[i, I[0]] = True
            else:
                n = int(len(I) / 2)
                _I = I[random.perm(len(I))[:n]]
                M[i, _I] = True

        _X = crossver_by_mask(X, M)
        return pop.new("X", _X)
