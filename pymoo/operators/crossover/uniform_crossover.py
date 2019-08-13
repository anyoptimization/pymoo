from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask
import numpy as np


class UniformCrossover(Crossover):

    def __init__(self, prob_uniform=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob_uniform = prob_uniform

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        # random matrix to do the crossover
        M = np.random.random((n_matings, n_var)) < self.prob_uniform

        _X = crossover_mask(X, M)
        return _X
