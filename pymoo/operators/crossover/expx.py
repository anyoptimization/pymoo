import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask, row_at_least_once_true


def mut_exp(n_matings, n_var, prob, at_least_once=True):

    # the mask do to the crossover
    M = np.full((n_matings, n_var), False)

    # start point of crossover
    s = np.random.randint(0, n_var, size=n_matings)

    # create for each individual the crossover range
    for i in range(n_matings):

        # the actual index where we start
        start = s[i]
        for j in range(n_var):

            # the current position where we are pointing to
            current = (start + j) % n_var

            # replace only if random value keeps being smaller than CR
            if np.random.random() <= prob:
                M[i, current] = True
            else:
                break

    if at_least_once:
        M = row_at_least_once_true(M)

    return M


class ExponentialCrossover(Crossover):

    def __init__(self, prob_exp=0.75, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob_exp = prob_exp

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = mut_exp(n_matings, n_var, self.prob_exp, at_least_once=True)
        _X = crossover_mask(X, M)
        return _X
