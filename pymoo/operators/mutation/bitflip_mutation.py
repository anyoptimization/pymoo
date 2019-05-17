import numpy as np

from pymoo.model.mutation import Mutation
from pymoo.rand import random


class BinaryBitflipMutation(Mutation):

    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, pop, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        X = pop.get("X")
        _X = np.full(X.shape, np.inf)

        M = random.random(X.shape)
        flip, not_flip = M < self.prob, M > self.prob

        _X[flip] = np.logical_not(X[flip])
        _X[not_flip] = X[not_flip]

        return pop.new("X", _X.astype(np.bool))
