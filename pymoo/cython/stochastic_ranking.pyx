# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True


import numpy as np


def stochastic_ranking(double[:] f, double[:] phi, double pr, long[:] I=None):
    if I is None:
        I = np.arange(len(f))
    return np.array(c_stochastic_ranking(f, phi, pr, I))



def c_stochastic_ranking(double[:] f, double[:] phi, double pr, long[:] I):
    cdef int _lambda, i, j, at_least_one_swap
    cdef double u

    _lambda = len(f)

    for i in range(_lambda):

        at_least_one_swap = 0

        for j in range(_lambda - 1):

            u = np.random.random()

            if u < pr or (phi[I[j]] == 0 and phi[I[j + 1]] == 0):
                if f[I[j]] > f[I[j + 1]]:
                    c_swap(I, j, j + 1)
                    at_least_one_swap = 1
            else:
                if phi[I[j]] > phi[I[j + 1]]:
                    c_swap(I, j, j + 1)
                    at_least_one_swap = 1

        if at_least_one_swap == 0:
            break

    return I


cdef c_swap(long[:] x, int a, int b):
    cdef int tmp
    tmp = x[a]
    x[a] = x[b]
    x[b] = tmp

