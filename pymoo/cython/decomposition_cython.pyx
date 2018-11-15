# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True


from calc_perpendicular_distance_cython cimport c_norm
from libcpp.vector cimport vector

import numpy as np


def pbi(double[:,:] F, double[:,:] weights, double[:] ideal_point, double theta, double eps=1e-10):
    return np.array(c_pbi(F, weights, ideal_point, theta, eps), dtype=np.float)


cdef extern from "math.h":
    double sqrt(double m)
    double pow(double base, double exponent)


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


