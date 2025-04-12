# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True


import numpy as np
from libcpp.vector cimport vector


# -----------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------

def calc_perpendicular_distance(double[:,:] P, double[:,:] L):
    return np.array(c_calc_perpendicular_distance(P, L))


def pbi(double[:,:] F, double[:,:] weights, double[:] ideal_point, double theta, double eps=1e-10):
    return np.array(c_pbi(F, weights, ideal_point, theta, eps), dtype=np.float64)


def calc_distance_to_weights(F, weights, utopian_point=None):

    if utopian_point is None:
        utopian_point = np.zeros(F.shape[1])

    norm = np.linalg.norm(weights, axis=1)

    d1, d2 = np.zeros(F.shape[0]), np.zeros(F.shape[0])

    F = F - utopian_point
    c_d1(d1, F, weights, norm)
    c_d2(d2, F, weights, d1, norm)

    return d1, d2


# -----------------------------------------------------------
# IMPLEMENTATION
# -----------------------------------------------------------

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



cdef vector[double] c_pbi(double[:,:] F, double[:,:] weights, double[:] ideal_point, double theta, double eps):
    cdef:
        double d1, d2, f_max, norm
        int i, j, n_dim
        vector[double] pbi

    n_points = F.shape[0]
    n_obj = F.shape[1]
    pbi = vector[double](n_points)

    for i in range(n_points):

        norm = c_norm(weights[i,:])

        d1 = 0
        for j in range(n_obj):
            d1 += (F[i,j] - ideal_point[j] + eps) * weights[i,j]
        d1 = d1 / norm

        d2 = 0
        for j in range(n_obj):
            d2 += pow(F[i,j] - ideal_point[j] + eps - (d1 * weights[i,j] / norm), 2.0)
        d2 = sqrt(d2)

        pbi[i] = d1 + theta * d2

    return pbi




cdef void c_d1(double[:] d1, double[:,:] F, double[:,:] weights, double[:] norm):
    cdef:
        double val
        int i, j

    n_points = F.shape[0]
    n_obj = F.shape[1]

    for i in range(n_points):

        val = 0
        for j in range(n_obj):
            val += F[i,j] * weights[i,j]
        d1[i] = val / norm[i]




cdef void c_d2(double[:] d2, double[:,:] F, double[:,:] weights, double[:] d1, double[:] norm):
    cdef:
        double val
        int i, j

    n_points = F.shape[0]
    n_obj = F.shape[1]

    for i in range(n_points):

        val = 0
        for j in range(n_obj):
            val += pow(F[i,j] - (d1[i] * weights[i,j] / norm[i]), 2.0)
        d2[i] = sqrt(val)




