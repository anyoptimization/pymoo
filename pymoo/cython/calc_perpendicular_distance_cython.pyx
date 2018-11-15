# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True


import numpy as np
from libcpp.vector cimport vector



def calc_perpendicular_distance(double[:,:] P, double[:,:] L):
    return np.array(c_calc_perpendicular_distance(P, L))


cdef extern from "math.h":
    double sqrt(double m)
    double pow(double base, double exponent)


cdef double c_norm(double[:] v):
    cdef:
        double val
        int i
    val = 0
    for i in range(v.shape[0]):
        val += pow(v[i], 2)
    val = sqrt(val)
    return val


cdef double[:,:] c_calc_perpendicular_distance(double[:,:] P, double[:,:] L):
    cdef :
        int s_L, s_P, n_dim, i, j, k
        double[:,:] M
        vector[double] N
        double norm, dot, perp_dist, norm_scalar_proj

    s_L = L.shape[0]
    s_P = P.shape[0]
    n_dim = L.shape[1]

    M = np.zeros((s_P, s_L), dtype=np.float64)

    for i in range(s_L):

        norm = c_norm(L[i, :])

        N = vector[double](n_dim)
        for k in range(n_dim):
            N[k] = L[i, k] / norm

        for j in range(s_P):

            dot = 0
            for k in range(n_dim):
                dot += L[i, k] * P[j, k]
            norm_scalar_proj = dot / norm

            perp_dist = 0
            for k in range(n_dim):
                perp_dist += pow(norm_scalar_proj * N[k] - P[j, k], 2)
            perp_dist = sqrt(perp_dist)

            M[j, i] = perp_dist

    return M


