import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask


class ExponentialCrossover(Crossover):

    def __init__(self, prob_exp=0.75, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob_exp = prob_exp

    def _do(self, problem, X, **kwargs):

        # get the X of parents and count the matings
        _, n_matings, n_var = X.shape

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # start point of crossover
        n = np.random.randint(0, n_var, size=X.shape[1])

        # the probabilities are calculated beforehand
        r = np.random.random((n_matings, n_var)) < self.prob_exp

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
        return _X
