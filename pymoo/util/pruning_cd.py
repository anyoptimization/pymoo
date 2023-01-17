import numpy as np

def calc_pcd(X, n_remove=0):

    N = X.shape[0]
    M = X.shape[1]

    if n_remove <= (N - M):
        if n_remove < 0:
            n_remove = 0
        else:
            pass
    else:
        n_remove = N - M

    extremes_min = np.argmin(X, axis=0)
    extremes_max = np.argmax(X, axis=0)
    
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    extremes = np.concatenate((extremes_min, extremes_max))
    
    X = (X - min_vals) / (max_vals - min_vals)
    
    H = np.arange(N)
    d = np.full(N, np.inf)
    
    I = np.argsort(X, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    _X = X[I, np.arange(M)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([_X, np.full(M, np.inf)]) - np.row_stack([np.full(M, -np.inf), _X])

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last, dist_to_next = dist_to_last[:-1], dist_to_next[1:]

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    _d = np.sum(dist_to_last[J, np.arange(M)] + dist_to_next[J, np.arange(M)], axis=1)
    d[H] = _d
    d[extremes] = np.inf
    
    n_removed = 0

    #While n_remove not acheived
    while n_removed < (n_remove - 1):

        #Obtain element to drop
        _d = d[H]
        _k = np.argmin(_d)
        k = H[_k]
        
        H = H[H != k]
        
        #Update index
        n_removed = n_removed + 1

        I = np.argsort(X[H].copy(), axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _X = X[H].copy()[I, np.arange(M)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_X, np.full(M, np.inf)]) - np.row_stack([np.full(M, -np.inf), _X])

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1], dist_to_next[1:]

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _d = np.sum(dist_to_last[J, np.arange(M)] + dist_to_next[J, np.arange(M)], axis=1)
        d[H] = _d
        d[extremes] = np.inf
    
    return d

