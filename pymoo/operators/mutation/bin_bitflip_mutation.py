from pymoo.model.mutation import Mutation
from pymoo.rand import random


class BinaryBitflipMutation(Mutation):
    def __init__(self, p_mut=None):
        self.p_mut = p_mut

    def _do(self, p, X, Y):

        if self.p_mut is None:
            self.p_mut = 1.0 / len(X)

        for i in range(X.shape[0]):

            for j in range(X.shape[1]):

                if random.random() < self.p_mut:
                    Y[i,j] = not X[i,j]
                else:
                    Y[i, j] = X[i, j]

        return Y
