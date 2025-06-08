# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

cdef extern from "vendor/hypervolume.cpp":
    double overmars_yap(double * points, double * referencePoint, unsigned noObjectives, unsigned noPoints);


def hv(ref_point, F):
    F = F[np.argsort(F[:, -1])]
    n, m = F.shape
    return c_hv(F, ref_point, m, n)


def c_hv(double[:,:] F, double[:] ref_point, m, n):
    return overmars_yap(&F[0, 0], &ref_point[0], m, n)
    # return fpli_hv(&F[0,0], m, n, &ref_point[0])
