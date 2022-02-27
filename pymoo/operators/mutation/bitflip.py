import numpy as np

from pymoo.core.mutation import VariableWiseMutation


class BitflipMutation(VariableWiseMutation):

    def _do(self, problem, X, params=None, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = np.random.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp


class BFM(BitflipMutation):
    pass
