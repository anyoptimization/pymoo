"""
Standard Python implementation of M-nearest neighbor calculations.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def calc_mnn(X, n_remove=0):
    """Calculate M-nearest neighbor distances."""
    return calc_mnn_base(X, n_remove=n_remove, twonn=False)


def calc_2nn(X, n_remove=0):
    """Calculate 2-nearest neighbor distances."""
    return calc_mnn_base(X, n_remove=n_remove, twonn=True)


def calc_mnn_base(X, n_remove=0, twonn=False):
    """Base function for M-nearest neighbor calculations."""
    N = X.shape[0]
    M = X.shape[1]
    
    if N <= M:
        return np.full(N, np.inf)

    if n_remove <= (N - M):
        if n_remove < 0:
            n_remove = 0
        else:
            pass
    else:
        n_remove = N - M
    
    if twonn:
        M = 2

    extremes_min = np.argmin(X, axis=0)
    extremes_max = np.argmax(X, axis=0)
    
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    extremes = np.concatenate((extremes_min, extremes_max))
    
    X = (X - min_vals) / (max_vals - min_vals)
    
    H = np.arange(N)
    
    D = squareform(pdist(X, metric="sqeuclidean"))
    Dnn = np.partition(D, range(1, M+1), axis=1)[:, 1:M+1]
    d = np.prod(Dnn, axis=1)
    d[extremes] = np.inf
    
    n_removed = 0

    # While n_remove not achieved
    while n_removed < (n_remove - 1):

        # Obtain element to drop
        _d = d[H]
        _k = np.argmin(_d)
        k = H[_k]
        H = H[H != k]

        # Update index
        n_removed = n_removed + 1
        if n_removed == n_remove:
            break

        else:

            D[:, k] = np.inf
            Dnn[H] = np.partition(D[H], range(1, M+1), axis=1)[:, 1:M+1]
            d[H] = np.prod(Dnn[H], axis=1)
            d[extremes] = np.inf
    
    return d