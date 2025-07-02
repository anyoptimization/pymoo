import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.variable import get, Real
from pymoo.util import default_random_state
from pymoo.util.misc import crossover_mask, row_at_least_once_true


@default_random_state
def mut_exp(n_matings, n_var, prob, at_least_once=True, random_state=None):
    assert len(prob) == n_matings

    # the mask do to the crossover
    M = np.full((n_matings, n_var), False)

    # start point of crossover
    s = random_state.integers(0, n_var, size=n_matings)

    # create for each individual the crossover range
    for i in range(n_matings):

        # the actual index where we start
        start = s[i]
        for j in range(n_var):

            # the current position where we are pointing to
            current = (start + j) % n_var

            # replace only if random value keeps being smaller than CR
            if random_state.random() <= prob[i]:
                M[i, current] = True
            else:
                break

    if at_least_once:
        M = row_at_least_once_true(M, random_state=random_state)

    return M


class ExponentialCrossover(Crossover):

    def __init__(self, prob_exp=0.75, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob_exp = Real(prob_exp, bounds=(0.5, 0.9), strict=(0.0, 1.0))

    def _do(self, _, X, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape
        prob_exp = get(self.prob_exp, size=n_matings)

        M = mut_exp(n_matings, n_var, prob_exp, at_least_once=True, random_state=random_state)
        _X = crossover_mask(X, M)
        return _X
