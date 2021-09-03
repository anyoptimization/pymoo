import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask


class PointCrossover(Crossover):

    def __init__(self, n_points, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points

    def _do(self, problem, X, **kwargs):

        # get the X of parents and count the matings
        _, n_matings, n_var = X.shape

        # start point of crossover
        r = np.row_stack([np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, n_var)])

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # create for each individual the crossover range
        for i in range(n_matings):

            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[i, a:b] = True
                j += 2

        _X = crossover_mask(X, M)

        return _X
