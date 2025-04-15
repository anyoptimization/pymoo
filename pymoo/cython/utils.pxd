# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set as cpp_set


cdef extern from "math.h":
    double HUGE_VAL


# Returns elements to remove based on crowding metric d and heap of remaining elements H
cdef inline int c_get_drop(double[:] d, cpp_set[int] H):

    cdef:
        int i, min_i
        double min_d

    min_d = HUGE_VAL
    min_i = 0

    for i in H:

        if d[i] <= min_d:
            min_d = d[i]
            min_i = i

    return min_i


# Returns vector of positions of minimum values along axis 0 of a 2d memoryview
cdef inline vector[int] c_get_argmin(double[:, :] X):

    cdef:
        int N, M, min_i, n, m
        double min_val
        vector[int] indexes

    N = X.shape[0]
    M = X.shape[1]

    indexes = vector[int]()

    for m in range(M):

        min_i = 0
        min_val = X[0, m]

        for n in range(N):

            if X[n, m] < min_val:

                min_i = n
                min_val = X[n, m]

        indexes.push_back(min_i)

    return indexes


# Returns vector of positions of maximum values along axis 0 of a 2d memoryview
cdef inline vector[int] c_get_argmax(double[:, :] X):

    cdef:
        int N, M, max_i, n, m
        double max_val
        vector[int] indexes

    N = X.shape[0]
    M = X.shape[1]

    indexes = vector[int]()

    for m in range(M):

        max_i = 0
        max_val = X[0, m]

        for n in range(N):

            if X[n, m] > max_val:

                max_i = n
                max_val = X[n, m]

        indexes.push_back(max_i)

    return indexes


# Performs normalization of a 2d memoryview
cdef inline double[:, :] c_normalize_array(double[:, :] X, vector[int] extremes_max, vector[int] extremes_min):

    cdef:
        int N = X.shape[0]
        int M = X.shape[1]
        int n, m, l, u
        double l_val, u_val, diff_val
        vector[double] min_vals, max_vals

    min_vals = vector[double]()
    max_vals = vector[double]()

    m = 0
    for u in extremes_max:
        u_val = X[u, m]
        max_vals.push_back(u_val)
        m = m + 1

    m = 0
    for l in extremes_min:
        l_val = X[l, m]
        min_vals.push_back(l_val)
        m = m + 1

    for m in range(M):

        diff_val = max_vals[m] - min_vals[m]
        if diff_val == 0.0:
            diff_val = 1.0

        for n in range(N):

            X[n, m] = (X[n, m] - min_vals[m]) / diff_val

    return X
