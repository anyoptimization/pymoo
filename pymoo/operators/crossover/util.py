import numpy as np


def crossover_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = np.copy(X)
    _X[0][M] = X[1][M]
    _X[1][M] = X[0][M]

    return _X
