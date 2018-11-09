# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True


from calc_perpendicular_distance_cython cimport c_norm
from libcpp.vector cimport vector

import numpy as np


def cython_pbi(double[:,:] F, double[:,:] weights, double[:] ideal_point, double theta):
    return np.array(c_pbi(F, weights, ideal_point, theta), dtype=np.float)



cdef extern from "math.h":
    double sqrt(double m)
    double pow(double base, double exponent)


cdef vector[double] c_pbi(double[:,:] F, double[:,:] weights, double[:] ideal_point, double theta):
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
            d1 += pow(((F[i,j] - ideal_point[j]) * weights[i,j]) / norm, 2.0)
        d1 = sqrt(d1)

        d2 = 0
        for j in range(n_obj):
            d2 += pow(F[i,j] - (ideal_point[j] - d1 * weights[i,j]), 2.0)
        d2 = sqrt(d2)

        pbi[i] = d1 + theta * d2

    return pbi


