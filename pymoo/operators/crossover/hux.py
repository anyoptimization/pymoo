import math

import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


class HalfUniformCrossover(Crossover):

    def __init__(self, prob_hux=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob_hux = prob_hux

    def _do(self, _, X, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        not_equal = X[0] != X[1]

        # create for each individual the crossover range
        for i in range(n_matings):
            I = np.where(not_equal[i])[0]

            n = math.ceil(len(I) / 2)
            if n > 0:
                _I = I[random_state.permutation(len(I))[:n]]
                M[i, _I] = True

        _X = crossover_mask(X, M)
        return _X


class HUX(HalfUniformCrossover):
    pass
