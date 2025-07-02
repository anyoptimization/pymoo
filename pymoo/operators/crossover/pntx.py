import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


class PointCrossover(Crossover):

    def __init__(self, n_points, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points

    def _do(self, _, X, random_state=None, **kwargs):

        # get the X of parents and count the matings
        _, n_matings, n_var = X.shape

        # start point of crossover
        r = np.vstack([random_state.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
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

        Xp = crossover_mask(X, M)

        return Xp


class SinglePointCrossover(PointCrossover):

    def __init__(self, **kwargs):
        super().__init__(n_points=1, **kwargs)


class TwoPointCrossover(PointCrossover):

    def __init__(self, **kwargs):
        super().__init__(n_points=2, **kwargs)
