import numpy as np

from pymoo.model.mutation import Mutation


class BinaryBitflipMutation(Mutation):

    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        X = X.astype(np.bool)
        _X = np.full(X.shape, np.inf)

        M = np.random.random(X.shape)
        flip, no_flip = M < self.prob, M >= self.prob

        _X[flip] = np.logical_not(X[flip])
        _X[no_flip] = X[no_flip]

        return _X.astype(np.bool)
