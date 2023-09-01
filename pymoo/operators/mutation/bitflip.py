import numpy as np
import pymoo
from pymoo.core.mutation import Mutation


class BitflipMutation(Mutation):

    def _do(self, problem, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = pymoo.PYMOO_PRNG.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp


class BFM(BitflipMutation):
    pass
