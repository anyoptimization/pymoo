from pymoo.model.mutation import Mutation
import numpy as np


class InversionMutation(Mutation):
    """
    Inversion mutation for permutation encoding.
    It will randomly choose a segment of a chromosome and reverse it's order.
    For example:
    [1, 2, 3, 4, 5] --> [1, 4, 3, 2, 5]
    """
    def __init__(self, prob=1):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i, x in enumerate(X):
            if np.random.random() > self.prob:
                # do not mutate
                Y[i] = x.copy()
            else:
                y = x.copy()
                start, end = np.sort(np.random.choice(problem.n_var, 2, replace=False))
                y[start:end + 1] = np.flip(y[start:end + 1])
                Y[i] = y
        return Y
