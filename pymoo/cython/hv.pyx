# distutils: language = c
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True


cdef extern from "vendor/hv.c":
    double fpli_hv(double *data, int d, int n, const double *ref)


def hv(F, ref_point):
    n, m = F.shape
    return c_hv(F, m, n, ref_point)


def c_hv(double[:,:] F, m, n, double[:] ref_point):
    return fpli_hv(&F[0,0], m, n, &ref_point[0])