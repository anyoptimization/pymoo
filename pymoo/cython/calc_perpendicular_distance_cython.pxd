# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True


cdef double c_norm(double[:] v)
cdef double[:,:] c_calc_perpendicular_distance(double[:,:] P, double[:,:] L)