import numpy as np


def crossover_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = np.copy(X)
    _X = _X.reshape(-1, _X.shape[-1])

    # first the second parent donors the to first
    _X[:len(M)][M] = X[1][M]

    # now the first parent donors to the second
    _X[len(M):][M] = X[0][M]

    return _X
