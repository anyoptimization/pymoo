# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

from pymoo.cython.utils cimport c_get_drop, c_get_argmin, c_get_argmax, c_normalize_array

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set as cpp_set


cdef extern from "math.h":
    double HUGE_VAL


# Python definition
def calc_pcd(double[:, :] X, int n_remove=0):

    cdef:
        int N, M, n
        cpp_set[int] extremes
        vector[int] extremes_min, extremes_max
        int[:, :] I

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

    _I = np.argsort(X, axis=0, kind='mergesort').astype(np.intc)
    I = _I[:, :]

    X = c_normalize_array(X, extremes_max, extremes_min)

    return c_calc_pcd(X, I, n_remove, N, M, extremes)


# Returns crowding metrics with recursive elimination
cdef c_calc_pcd(double[:, :] X, int[:, :] I, int n_remove, int N, int M, cpp_set[int] extremes):

    cdef:
        int n, n_removed, k
        cpp_set[int] calc_items
        cpp_set[int] H
        double[:, :] D
        double[:] d

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

    # Initialize
    n_removed = 0

    # Initialize neighbors and distances
    _D = np.full((N, M), HUGE_VAL, dtype=np.double)
    dd = np.full((N,), HUGE_VAL, dtype=np.double)

    D = _D[:, :]
    d = dd[:]

    # Fill in neighbors and distance matrix
    c_calc_pcd_iter(
            X,
            I,
            D,
            N, M,
            calc_items,
        )

    # Obtain distance metrics
    c_calc_d(d, D, calc_items, M)

    # While n_remove not acheived
    while n_removed < (n_remove - 1):

        # Obtain element to drop
        k = c_get_drop(d, H)
        H.erase(k)

        # Update index
        n_removed = n_removed + 1

        # Get items to be recalculated
        calc_items = c_get_calc_items(I, k, M, N)
        for n in extremes:
            calc_items.erase(n)

        # Fill in neighbors and distance matrix
        c_calc_pcd_iter(
                X,
                I,
                D,
                N, M,
                calc_items,
            )

        # Obtain distance metrics
        c_calc_d(d, D, calc_items, M)

    return dd


# Iterate
cdef c_calc_pcd_iter(
    double[:, :] X,
    int[:, :] I,
    double[:, :] D,
    int N, int M,
    cpp_set[int] calc_items,
    ):

    cdef:
        int i, m, n, l, u

    # Iterate over items to calculate
    for i in calc_items:

        # Iterate over elements in X
        for m in range(M):

            for n in range(N):

                if i == I[n, m]:

                    l = I[n - 1, m]
                    u = I[n + 1, m]

                    D[i, m] = (X[u, m] - X[l, m]) / M


# Calculate crowding metric
cdef c_calc_d(double[:] d, double[:, :] D, cpp_set[int] calc_items, int M):

    cdef:
        int i, m

    for i in calc_items:

        d[i] = 0
        for m in range(M):
            d[i] = d[i] + D[i, m]


# Returns indexes of items to be recalculated after removal
cdef cpp_set[int] c_get_calc_items(
    int[:, :] I,
    int k, int M, int N
    ):

    cdef:
        int n, m
        cpp_set[int] calc_items

    calc_items = cpp_set[int]()

    # Iterate over all elements in I
    for m in range(M):

        for n in range(N):

            if I[n, m] == k:

                # Add to set of items to be recalculated
                calc_items.insert(I[n - 1, m])
                calc_items.insert(I[n + 1, m])

                # Remove element from sorted array
                I[n:-1, m] = I[n + 1:, m]

    return calc_items
