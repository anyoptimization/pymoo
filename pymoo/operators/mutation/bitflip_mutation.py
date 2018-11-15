import numpy as np

from pymoo.model.mutation import Mutation
from pymoo.rand import random


class BinaryBitflipMutation(Mutation):

    def __init__(self, p_mut=None):
        super().__init__()
        self.p_mut = p_mut

    def _do(self, problem, pop, **kwargs):
        if self.p_mut is None:
            self.p_mut = 1.0 / problem.n_var

        X = pop.get("X")
        _X = np.full(X.shape, np.inf)

        M = random.random(X.shape)
        flip, not_flip = M < self.p_mut, M > self.p_mut

        _X[flip] = np.logical_not(X[flip])
        _X[not_flip] = X[not_flip]

        return pop.new("X", _X.astype(np.bool))
