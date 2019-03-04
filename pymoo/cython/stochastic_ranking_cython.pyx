# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

from libc.stdlib cimport rand
cdef extern from "limits.h":
    int INT_MAX



# swap two individuals in the _current population
cdef func_swap(long[:] A, int i, int j):
    cdef int tmp = A[i]
    A[i] = A[j]
    A[j] = tmp

def stochastic_ranking(double[:] F, double[:] CV, double prob):

    cdef:
        int n_solutions, _lambda, swap, _current, _next
        int i, j
        long[:] index

    # the number of solutions that need to be ranked
    n_solutions = F.shape[0]

    # the number of pairwise comparisons - here we fix it to the number of solutions
    _lambda = n_solutions

    # the final sorting
    index = np.arange(n_solutions)

    for i in range(n_solutions):

        # variable which sets the flag if a swap was performed or not
        swap = 0

        for j in range(_lambda - 1):

            _current, _next = index[j], index[j + 1]

            if (CV[_current] == 0 and CV[_next] == 0) or (rand() / float(INT_MAX) < prob):

                if F[_current] > F[_next]:
                    func_swap(index, j, j + 1)
                    swap = 1

            else:

                if CV[_current] > CV[_next]:
                    func_swap(index, j, j + 1)
                    swap = 1

        if swap == 0:
            break

    return index

