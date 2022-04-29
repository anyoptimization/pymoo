import math

import numpy as np

from pymoo.core.selection import Selection
from pymoo.util.misc import random_permuations


class RandomSelection(Selection):

    def _do(self, _, pop, n_select, n_parents, **kwargs):
        # number of random individuals needed
        n_random = n_select * n_parents

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop))[:n_random]

        return np.reshape(P, (n_select, n_parents))


def fast_fill_random(X, N, columns=None, Xp=None, n_max_attempts=10):
    """

    Parameters
    ----------
    X : np.ndarray
        The actually array to fill with random values.
    N : int
        The upper limit for the values. The values will be in range (0, ..., N)
    columns : list
        The columns which should be filled randomly. Other columns indicate duplicates
    Xp : np.ndarray
        If some other duplicates shall be avoided by default

    """

    _, n_cols = X.shape

    if columns is None:
        columns = range(n_cols)

    # all columns set so far to be checked for duplicates
    J = []

    # for each of the columns which should be set to be no duplicates
    for col in columns:

        D = X[:, J]
        if Xp is not None:
            D = np.column_stack([D, Xp])

        # all the remaining indices that need to be filled with no duplicates
        rem = np.arange(len(X))

        for _ in range(n_max_attempts):

            if len(rem) > N:
                X[rem, col] = np.random.choice(N, replace=True, size=len(rem))
            else:
                X[rem, col] = np.random.permutation(N)[:len(rem)]

            rem = np.where((X[rem, col][:, None] == D[rem]).any(axis=1))[0]

            if len(rem) == 0:
                break

        J.append(col)

    return X
