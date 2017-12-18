import numpy as np

from pymoo.model.mutation import Mutation


class BinaryBitflipMutation(Mutation):
    def __init__(self, p_mut=None):
        self.p_mut = p_mut

    def _do(self, p, x):

        if self.p_mut is None:
            self.p_mut = 1.0 / len(x)

        for i in range(p.n_var):
            if np.random.rand() < self.p_mut:
                x[i] = not x[i]
