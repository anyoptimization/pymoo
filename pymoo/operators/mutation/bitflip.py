import numpy as np

from pymoo.core.mutation import Mutation


class BitflipMutation(Mutation):

    def _do(self, problem, X, random_state=None, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = random_state.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp


class BFM(BitflipMutation):
    pass
