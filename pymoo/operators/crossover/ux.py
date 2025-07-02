import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


class UniformCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape
        M = random_state.random((n_matings, n_var)) < 0.5
        _X = crossover_mask(X, M)
        return _X


class UX(UniformCrossover):
    pass
