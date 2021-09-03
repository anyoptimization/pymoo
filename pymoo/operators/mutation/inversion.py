from pymoo.model.mutation import Mutation
import numpy as np

from pymoo.operators.crossover.order_crossover import random_sequence


def inversion_mutation(y, seq, inplace=True):
    y = y if inplace else np.copy(y)

    seq = seq if not None else random_sequence(len(y))
    start, end = seq

    y[start:end + 1] = np.flip(y[start:end + 1])
    return y


class InversionMutation(Mutation):

    def __init__(self, prob=1.0):
        """

        This mutation is applied to permutations. It randomly selects a segment of a chromosome and reverse its order.
        For instance, for the permutation `[1, 2, 3, 4, 5]` the segment can be `[2, 3, 4]` which results in `[1, 4, 3, 2, 5]`.

        Parameters
        ----------
        prob : float
            Probability to apply the mutation to the individual
            
        """
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i, y in enumerate(X):
            if np.random.random() < self.prob:
                seq = random_sequence(len(y))
                Y[i] = inversion_mutation(y, seq, inplace=True)

        return Y
