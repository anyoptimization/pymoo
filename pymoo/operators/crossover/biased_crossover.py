import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask


class BiasedCrossover(Crossover):

    def __init__(self, bias, **kwargs):
        super().__init__(2, 1, **kwargs)
        self.bias = bias

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < self.bias
        _X = crossover_mask(X, M)
        return _X

    def do(self, problem, pop, parents, **kwargs):
        off = super().do(problem, pop, parents, **kwargs)
        return off[:int(len(off) / 2)]
