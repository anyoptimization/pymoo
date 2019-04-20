import numpy as np


def crossver_by_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = X.reshape(-1, X.shape[-1])

    # invert the whole logical array
    _M = np.logical_not(M)

    # first the second parent donors the to first
    _X[:len(M)][M] = X[1][M]

    # now the first parent donors to the second
    _X[len(M):][_M] = X[0][_M]

    return _X
