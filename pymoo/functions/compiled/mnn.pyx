# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

# This was implemented using the full distances matrix
# Other strategies can be more efficient depending on the population size and number of objectives
# This approach was the most promising for N = 3
# I believe for a large number of objectives M, some strategy based on upper bounds for distances would be helpful
# Those interested in contributing please contact me at bruscalia12@gmail.com


import numpy as np

from pymoo.functions.compiled.utils cimport c_get_drop, c_get_argmin, c_get_argmax, c_normalize_array

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set as cpp_set


cdef extern from "math.h":
    double HUGE_VAL


def calc_mnn(double[:, :] X, int n_remove=0):

    cdef:
        int N, M, n
        cpp_set[int] extremes
        vector[int] extremes_min, extremes_max

    N = X.shape[0]
    M = X.shape[1]

    if N <= M:
        return np.full(N, HUGE_VAL)

    if n_remove <= (N - M):
        if n_remove < 0:
            n_remove = 0
        else:
            pass
    else:
        n_remove = N - M

    extremes_min = c_get_argmin(X)
    extremes_max = c_get_argmax(X)

    extremes = cpp_set[int]()

    for n in extremes_min:
        extremes.insert(n)

    for n in extremes_max:
        extremes.insert(n)

    X = c_normalize_array(X, extremes_max, extremes_min)

    return c_calc_mnn(X, n_remove, N, M, extremes)


def calc_2nn(double[:, :] X, int n_remove=0):

    cdef:
        int N, M, n
        cpp_set[int] extremes
        vector[int] extremes_min, extremes_max

    N = X.shape[0]
    M = X.shape[1]

    if n_remove <= (N - M):
        if n_remove < 0:
            n_remove = 0
        else:
            pass
    else:
        n_remove = N - M

    extremes_min = c_get_argmin(X)
    extremes_max = c_get_argmax(X)

    extremes = cpp_set[int]()

    for n in extremes_min:
        extremes.insert(n)

    for n in extremes_max:
        extremes.insert(n)

    X = c_normalize_array(X, extremes_max, extremes_min)

    M = 2

    return c_calc_mnn(X, n_remove, N, M, extremes)


cdef c_calc_mnn(double[:, :] X, int n_remove, int N, int M, cpp_set[int] extremes):

    cdef:
        int n, mm, i, j, n_removed, k, MM
        double dij
        cpp_set[int] calc_items
        cpp_set[int] H
        double[:, :] D
        double[:] d
        int[:, :] Mnn

    # Define items to calculate distances
    calc_items = cpp_set[int]()
    for n in range(N):
        calc_items.insert(n)
    for n in extremes:
        calc_items.erase(n)

    # Define remaining items to evaluate
    H = cpp_set[int]()
    for n in range(N):
        H.insert(n)

    # Instantiate distances array
    _D = np.empty((N, N), dtype=np.double)
    D = _D[:, :]

    # Shape of X
    MM = X.shape[1]

    # Fill values on D
    for i in range(N - 1):
        D[i, i] = 0.0

        for j in range(i + 1, N):

            dij = 0
            for mm in range(MM):
                dij = dij + (X[j, mm] - X[i, mm]) * (X[j, mm] - X[i, mm])

            D[i, j] = dij
            D[j, i] = D[i, j]

    D[N-1, N-1] = 0.0

    # Initialize
    n_removed = 0

    # Initialize neighbors and distances
    # _Mnn = np.full((N, M), -1, dtype=np.intc)
    _Mnn = np.argpartition(D, range(1, M+1), axis=1)[:, 1:M+1].astype(np.intc)
    dd = np.full((N,), HUGE_VAL, dtype=np.double)

    Mnn = _Mnn[:, :]
    d = dd[:]

    # Obtain distance metrics
    c_calc_d(d, Mnn, D, calc_items, M)

    # While n_remove not acheived (no need to recalculate if only one item should be removed)
    while n_removed < (n_remove - 1):

        # Obtain element to drop
        k = c_get_drop(d, H)
        H.erase(k)

        # Update index
        n_removed = n_removed + 1

        # Get items to be recalculated
        calc_items = c_get_calc_items(Mnn, H, k, M)
        for n in extremes:
            calc_items.erase(n)

        # Fill in neighbors and distance matrix
        c_calc_mnn_iter(
                X,
                Mnn,
                D,
                N, M,
                calc_items,
                H
            )

        # Obtain distance metrics
        c_calc_d(d, Mnn, D, calc_items, M)

    return dd


cdef c_calc_mnn_iter(
    double[:, :] X,
    int[:, :] Mnn,
    double[:, :] D,
    int N, int M,
    cpp_set[int] calc_items,
    cpp_set[int] H
    ):

    cdef:
        int i, j, m

    # Iterate over items to calculate
    for i in calc_items:

        # Iterate over elements in X
        for j in H:

            # Go to next if same element
            if (j == i):
                continue

            # Replace at least the last neighbor
            elif (D[i, j] <= D[i, Mnn[i, M-1]]) or (Mnn[i, M-1] == -1):

                # Iterate over current values
                for m in range(M):

                    # Set to current if unassigned
                    if (Mnn[i, m] == -1):

                        # Set last neighbor to index
                        Mnn[i, m] = j
                        break

                    # Break if checking already corresponding index
                    elif (j == Mnn[i, m]):
                        break

                    # Distance satisfies condition
                    elif (D[i, j] <= D[i, Mnn[i, m]]):

                        # Replace higher values
                        Mnn[i, m + 1:] = Mnn[i, m:-1]

                        # Replace current value
                        Mnn[i, m] = j
                        break


# Calculate crowding metric
cdef c_calc_d(double[:] d, int[:, :] Mnn, double[:, :] D, cpp_set[int] calc_items, int M):

    cdef:
        int i, m

    for i in calc_items:

        d[i] = 1
        for m in range(M):
            d[i] = d[i] * D[i, Mnn[i, m]]


# Returns indexes of items to be recalculated after removal
cdef cpp_set[int] c_get_calc_items(
    int[:, :] Mnn,
    cpp_set[int] H,
    int k, int M):

    cdef:
        int i, m
        cpp_set[int] calc_items

    calc_items = cpp_set[int]()

    for i in H:

        for m in range(M):

            if Mnn[i, m] == k:

                Mnn[i, m:-1] = Mnn[i, m + 1:]
                Mnn[i, M-1] = -1

                calc_items.insert(i)

    return calc_items
